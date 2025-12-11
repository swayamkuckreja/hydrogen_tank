# Navier-Stokes with passive temperature scalar
using Ferrite, SparseArrays, BlockArrays, LinearAlgebra, UnPack, LinearSolve, WriteVTK
using OrdinaryDiffEq
using DiffEqBase

ν = 1.0 / 1000.0  # kinematic viscosity
κ = 1.0e-4        # thermal diffusivity

using FerriteGmsh
using FerriteGmsh: Gmsh
Gmsh.initialize()
gmsh.option.set_number("General.Verbosity", 2)
dim = 2

rect_tag = gmsh.model.occ.add_rectangle(0, 0, 0, 1.1, 0.41)
circle_tag = gmsh.model.occ.add_circle(0.2, 0.2, 0, 0.05)
circle_curve_tag = gmsh.model.occ.add_curve_loop([circle_tag])
circle_surf_tag = gmsh.model.occ.add_plane_surface([circle_curve_tag])
gmsh.model.occ.cut([(dim, rect_tag)], [(dim, circle_surf_tag)])

gmsh.model.occ.synchronize()

bottomtag = gmsh.model.model.add_physical_group(dim - 1, [6], -1, "bottom")
lefttag = gmsh.model.model.add_physical_group(dim - 1, [7], -1, "left")
righttag = gmsh.model.model.add_physical_group(dim - 1, [8], -1, "right")
toptag = gmsh.model.model.add_physical_group(dim - 1, [9], -1, "top")
holetag = gmsh.model.model.add_physical_group(dim - 1, [5], -1, "hole")

gmsh.option.setNumber("Mesh.Algorithm", 11)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)

gmsh.model.mesh.generate(dim)
grid = togrid()
Gmsh.finalize()

# Finite element spaces
ip_v = Lagrange{RefQuadrilateral, 2}()^2  # velocity Q2 (2D vector)
ip_p = Lagrange{RefQuadrilateral, 1}()     # pressure Q1
ip_T = Lagrange{RefQuadrilateral, 1}()     # temperature Q1

qr = QuadratureRule{RefQuadrilateral}(4)
cellvalues_v = CellValues(qr, ip_v)
cellvalues_p = CellValues(qr, ip_p)
cellvalues_T = CellValues(qr, ip_T)

# DofHandler with three fields: v, p, T
dh = DofHandler(grid)
add!(dh, :v, ip_v)
add!(dh, :p, ip_p)
add!(dh, :T, ip_T)
close!(dh)

# Boundary conditions
ch = ConstraintHandler(dh)

# No-slip on walls and cylinder
nosplip_facet_names = ["top", "bottom", "hole"]
∂Ω_noslip = union(getfacetset.((grid,), nosplip_facet_names)...)
noslip_bc = Dirichlet(:v, ∂Ω_noslip, (x, t) -> Vec((0.0, 0.0)), [1, 2])
add!(ch, noslip_bc)

# Inflow velocity profile
∂Ω_inflow = getfacetset(grid, "left")
vᵢₙ(t) = min(t * 1.5, 1.5)
parabolic_inflow_profile(x, t) = Vec((4 * vᵢₙ(t) * x[2] * (0.41 - x[2]) / 0.41^2, 0.0))
inflow_bc = Dirichlet(:v, ∂Ω_inflow, parabolic_inflow_profile, [1, 2])
add!(ch, inflow_bc)

# Temperature BC at inlet
T_inlet(x, t) = 1.0
temp_bc = Dirichlet(:T, ∂Ω_inflow, (x, t) -> T_inlet(x, t), [1])
add!(ch, temp_bc)

close!(ch)
update!(ch, 0.0)

# Mass matrix assembly (v and T have time derivatives, p does not)
function assemble_mass_matrix(cellvalues_v, cellvalues_p, cellvalues_T, M::SparseMatrixCSC, dh::DofHandler)
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs_T = getnbasefunctions(cellvalues_T)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p + n_basefuncs_T
    v▄, p▄, T▄ = 1, 2, 3
    Mₑ = BlockedArray(zeros(n_basefuncs, n_basefuncs), 
                      [n_basefuncs_v, n_basefuncs_p, n_basefuncs_T], 
                      [n_basefuncs_v, n_basefuncs_p, n_basefuncs_T])

    mass_assembler = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(Mₑ, 0)
        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_T, cell)

        # Velocity mass block (1,1)
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            for i in 1:n_basefuncs_v
                φᵢ = shape_value(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    φⱼ = shape_value(cellvalues_v, q_point, j)
                    Mₑ[BlockIndex((v▄, v▄), (i, j))] += φᵢ ⋅ φⱼ * dΩ
                end
            end
        end

        # Temperature mass block (3,3)
        for q_point in 1:getnquadpoints(cellvalues_T)
            dΩ = getdetJdV(cellvalues_T, q_point)
            for i in 1:n_basefuncs_T
                φᵢ = shape_value(cellvalues_T, q_point, i)
                for j in 1:n_basefuncs_T
                    φⱼ = shape_value(cellvalues_T, q_point, j)
                    Mₑ[BlockIndex((T▄, T▄), (i, j))] += φᵢ * φⱼ * dΩ
                end
            end
        end

        assemble!(mass_assembler, celldofs(cell), Mₑ)
    end
    return M
end

# Stiffness matrix assembly (Stokes operator for v,p + diffusion for T)
function assemble_stiffness_matrix(cellvalues_v, cellvalues_p, cellvalues_T, ν, κ, K::SparseMatrixCSC, dh::DofHandler)
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs_T = getnbasefunctions(cellvalues_T)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p + n_basefuncs_T
    v▄, p▄, T▄ = 1, 2, 3
    Kₑ = BlockedArray(zeros(n_basefuncs, n_basefuncs), 
                      [n_basefuncs_v, n_basefuncs_p, n_basefuncs_T], 
                      [n_basefuncs_v, n_basefuncs_p, n_basefuncs_T])

    stiffness_assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Kₑ, 0)
        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_p, cell)
        Ferrite.reinit!(cellvalues_T, cell)

        # Velocity-velocity block (1,1): viscous term
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            for i in 1:n_basefuncs_v
                ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    ∇φⱼ = shape_gradient(cellvalues_v, q_point, j)
                    Kₑ[BlockIndex((v▄, v▄), (i, j))] -= ν * ∇φᵢ ⊡ ∇φⱼ * dΩ
                end
            end
        end

        # Velocity-pressure coupling blocks (1,2) and (2,1)
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            for j in 1:n_basefuncs_p
                ψ = shape_value(cellvalues_p, q_point, j)
                for i in 1:n_basefuncs_v
                    divφ = shape_divergence(cellvalues_v, q_point, i)
                    Kₑ[BlockIndex((v▄, p▄), (i, j))] += (divφ * ψ) * dΩ
                    Kₑ[BlockIndex((p▄, v▄), (j, i))] += (ψ * divφ) * dΩ
                end
            end
        end

        # Temperature-temperature block (3,3): thermal diffusion
        for q_point in 1:getnquadpoints(cellvalues_T)
            dΩ = getdetJdV(cellvalues_T, q_point)
            for i in 1:n_basefuncs_T
                ∇φᵢ = shape_gradient(cellvalues_T, q_point, i)
                for j in 1:n_basefuncs_T
                    ∇φⱼ = shape_gradient(cellvalues_T, q_point, j)
                    Kₑ[BlockIndex((T▄, T▄), (i, j))] -= κ * (∇φᵢ ⋅ ∇φⱼ) * dΩ
                end
            end
        end

        assemble!(stiffness_assembler, celldofs(cell), Kₑ)
    end
    return K
end

# Allocate and assemble matrices
M = allocate_matrix(dh)
M = assemble_mass_matrix(cellvalues_v, cellvalues_p, cellvalues_T, M, dh)

K = allocate_matrix(dh)
K = assemble_stiffness_matrix(cellvalues_v, cellvalues_p, cellvalues_T, ν, κ, K, dh)

u₀ = zeros(ndofs(dh))
apply!(u₀, ch)

jac_sparsity = sparse(K)

apply!(M, ch)

# Parameters for RHS
struct RHSparams
    K::SparseMatrixCSC
    ch::ConstraintHandler
    dh::DofHandler
    cellvalues_v::CellValues
    cellvalues_T::CellValues
    u::Vector
end
p = RHSparams(K, ch, dh, cellvalues_v, cellvalues_T, copy(u₀))

function ferrite_limiter!(u, _, p, t)
    update!(p.ch, t)
    return apply!(u, p.ch)
end

# Nonlinear convection for velocity
function navierstokes_rhs_element!(dvₑ, vₑ, cellvalues_v)
    n_basefuncs = getnbasefunctions(cellvalues_v)
    for q_point in 1:getnquadpoints(cellvalues_v)
        dΩ = getdetJdV(cellvalues_v, q_point)
        ∇v = function_gradient(cellvalues_v, q_point, vₑ)
        v = function_value(cellvalues_v, q_point, vₑ)
        for j in 1:n_basefuncs
            φⱼ = shape_value(cellvalues_v, q_point, j)
            dvₑ[j] -= v ⋅ ∇v' ⋅ φⱼ * dΩ
        end
    end
    return
end

# Temperature advection (passive scalar)
function temp_advection_element!(dTₑ, Tₑ, vₑ, cellvalues_T, cellvalues_v)
    n_basefuncs = getnbasefunctions(cellvalues_T)
    for q_point in 1:getnquadpoints(cellvalues_T)
        dΩ = getdetJdV(cellvalues_T, q_point)
        ∇T = function_gradient(cellvalues_T, q_point, Tₑ)
        v = function_value(cellvalues_v, q_point, vₑ)
        for j in 1:n_basefuncs
            φⱼ = shape_value(cellvalues_T, q_point, j)
            dTₑ[j] -= (v ⋅ ∇T) * φⱼ * dΩ
        end
    end
    return
end

# Combined RHS
function navierstokes_temp!(du, u_uc, p::RHSparams, t)
    @unpack K, ch, dh, cellvalues_v, cellvalues_T, u = p

    u .= u_uc
    update!(ch, t)
    apply!(u, ch)

    # Linear contribution: du = K * u
    mul!(du, K, u)

    # Nonlinear velocity convection
    v_range = dof_range(dh, :v)
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    vₑ = zeros(n_basefuncs_v)
    duₑ = zeros(n_basefuncs_v)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_v, cell)
        v_celldofs = @view celldofs(cell)[v_range]
        vₑ .= @views u[v_celldofs]
        fill!(duₑ, 0.0)
        navierstokes_rhs_element!(duₑ, vₑ, cellvalues_v)
        assemble!(du, v_celldofs, duₑ)
    end

    # Temperature advection
    T_range = dof_range(dh, :T)
    n_basefuncs_T = getnbasefunctions(cellvalues_T)
    Tₑ = zeros(n_basefuncs_T)
    dTₑ = zeros(n_basefuncs_T)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_T, cell)
        Ferrite.reinit!(cellvalues_v, cell)
        celld = celldofs(cell)
        T_celldofs = @view celld[T_range]
        v_celldofs = @view celld[v_range]
        Tₑ .= @views u[T_celldofs]
        vₑ .= @views u[v_celldofs]
        fill!(dTₑ, 0.0)
        temp_advection_element!(dTₑ, Tₑ, vₑ, cellvalues_T, cellvalues_v)
        assemble!(du, T_celldofs, dTₑ)
    end

    return
end

# Jacobian for velocity
function navierstokes_jac_element!(Jₑ, vₑ, cellvalues_v)
    n_basefuncs = getnbasefunctions(cellvalues_v)
    for q_point in 1:getnquadpoints(cellvalues_v)
        dΩ = getdetJdV(cellvalues_v, q_point)
        ∇v = function_gradient(cellvalues_v, q_point, vₑ)
        v = function_value(cellvalues_v, q_point, vₑ)
        for j in 1:n_basefuncs
            φⱼ = shape_value(cellvalues_v, q_point, j)
            for i in 1:n_basefuncs
                φᵢ = shape_value(cellvalues_v, q_point, i)
                ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
                Jₑ[j, i] -= (φᵢ ⋅ ∇v' + v ⋅ ∇φᵢ') ⋅ φⱼ * dΩ
            end
        end
    end
    return
end

# Jacobian for temperature advection
function temp_jac_element!(Jₑ, vₑ, cellvalues_T, cellvalues_v)
    n_basefuncs = getnbasefunctions(cellvalues_T)
    for q_point in 1:getnquadpoints(cellvalues_T)
        dΩ = getdetJdV(cellvalues_T, q_point)
        v = function_value(cellvalues_v, q_point, vₑ)
        for i in 1:n_basefuncs
            φᵢ = shape_value(cellvalues_T, q_point, i)
            for j in 1:n_basefuncs
                ∇φⱼ = shape_gradient(cellvalues_T, q_point, j)
                Jₑ[i, j] -= (φᵢ * (v ⋅ ∇φⱼ)) * dΩ
            end
        end
    end
    return
end

# Combined Jacobian
function navierstokes_temp_jac!(J, u_uc, p, t)
    @unpack K, ch, dh, cellvalues_v, cellvalues_T, u = p

    u .= u_uc
    update!(ch, t)
    apply!(u, ch)

    # Start from K
    nonzeros(J) .= nonzeros(K)

    assembler = start_assemble(J; fillzero = false)

    # Velocity Jacobian
    v_range = dof_range(dh, :v)
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    Jₑ = zeros(n_basefuncs_v, n_basefuncs_v)
    vₑ = zeros(n_basefuncs_v)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_v, cell)
        v_celldofs = @view celldofs(cell)[v_range]
        vₑ .= @views u[v_celldofs]
        fill!(Jₑ, 0.0)
        navierstokes_jac_element!(Jₑ, vₑ, cellvalues_v)
        assemble!(assembler, v_celldofs, Jₑ)
    end

    # Temperature Jacobian
    T_range = dof_range(dh, :T)
    n_basefuncs_T = getnbasefunctions(cellvalues_T)
    JTₑ = zeros(n_basefuncs_T, n_basefuncs_T)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_T, cell)
        Ferrite.reinit!(cellvalues_v, cell)
        celld = celldofs(cell)
        T_celldofs = @view celld[T_range]
        v_celldofs = @view celld[v_range]
        vₑ .= @views u[v_celldofs]
        fill!(JTₑ, 0.0)
        temp_jac_element!(JTₑ, vₑ, cellvalues_T, cellvalues_v)
        assemble!(assembler, T_celldofs, JTₑ)
    end

    return apply!(J, ch)
end

# ODE setup
rhs = ODEFunction(navierstokes_temp!, mass_matrix = M; jac = navierstokes_temp_jac!, jac_prototype = jac_sparsity)
problem = ODEProblem(rhs, u₀, (0.0, 6.0), p)

struct FreeDofErrorNorm
    ch::ConstraintHandler
end
(fe_norm::FreeDofErrorNorm)(u::Union{AbstractFloat, Complex}, t) = DiffEqBase.ODE_DEFAULT_NORM(u, t)
(fe_norm::FreeDofErrorNorm)(u::AbstractArray, t) = DiffEqBase.ODE_DEFAULT_NORM(u[fe_norm.ch.free_dofs], t)

timestepper = Rodas5P(autodiff = false, step_limiter! = ferrite_limiter!)

integrator = init(
    problem, timestepper; initializealg = NoInit(), dt = 0.001,
    adaptive = true, abstol = 1.0e-4, reltol = 1.0e-5,
    progress = true, progress_steps = 1,
    verbose = true, internalnorm = FreeDofErrorNorm(ch), d_discontinuities = [1.0]
)

# VTK output
pvd = paraview_collection("vortex-street-with-temp")
for (step, (u, t)) in enumerate(intervals(integrator))
    println("Step $step, t = $t, max|u| = $(maximum(abs, u))")
    VTKGridFile("vortex-street-with-temp-$step", dh) do vtk
        write_solution(vtk, dh, u)
        pvd[t] = vtk
    end
end
vtk_save(pvd)
println("Simulation complete!")
