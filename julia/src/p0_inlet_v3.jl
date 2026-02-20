
println("Starting execution...")
using Ferrite, SparseArrays, LinearAlgebra, UnPack, LinearSolve, WriteVTK, BlockArrays
using OrdinaryDiffEq, DiffEqBase
using FerriteGmsh
using FerriteGmsh: Gmsh
println("Dependencies loaded!")

# Meshing and geometry
dim = 2
Gmsh.initialize()
gmsh.option.set_number("General.Verbosity", 2)
rect_tag = gmsh.model.occ.add_rectangle(0, 0, 0, 1.1, 0.41)
circle_tag = gmsh.model.occ.add_circle(0.2, 0.2, 0, 0.05)
gmsh.model.occ.cut([(dim, rect_tag)], [(dim, gmsh.model.occ.add_plane_surface([gmsh.model.occ.add_curve_loop([circle_tag])]))])
gmsh.model.occ.synchronize()

gmsh.model.model.add_physical_group(dim - 1, [6], -1, "bottom")
gmsh.model.model.add_physical_group(dim - 1, [7], -1, "left")
gmsh.model.model.add_physical_group(dim - 1, [8], -1, "right")
gmsh.model.model.add_physical_group(dim - 1, [9], -1, "top")
gmsh.model.model.add_physical_group(dim - 1, [5], -1, "hole")

gmsh.option.setNumber("Mesh.Algorithm", 11)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)
gmsh.model.mesh.generate(dim)
grid = togrid()
Gmsh.finalize()

# Interpolation and quadrature rules 
ipu = Lagrange{RefQuadrilateral, 2}()^dim 
ipp = Lagrange{RefQuadrilateral, 1}()
ipg = Lagrange{RefQuadrilateral, 1}() # linear geometric interpolation

# Cell values
qr = QuadratureRule{RefQuadrilateral}(4)
cvu = CellValues(qr, ipu, ipg)
cvp = CellValues(qr, ipp, ipg)

# Facet values for nonlinear BC
tol_in = 0.0011
H = 1.0
qr_facet = FacetQuadratureRule{RefQuadrilateral}(4)
fvu = FacetValues(qr_facet, ipu, ipg)
fvp = FacetValues(qr_facet, ipp, ipg)

# Dof handler : velocity ux uy and pressure p 
dh = DofHandler(grid)
add!(dh, :u, ipu)
add!(dh, :p, ipp)
close!(dh)

# Boundary conditions
dbc = Dirichlet(:u, union(getfacetset(dh.grid, "top"), getfacetset(dh.grid, "bottom"), getfacetset(dh.grid, "hole")), (x, t) -> [0.0, 0.0], [1,2])
dbc_outlet = Dirichlet(:p, getfacetset(dh.grid, "right"), (x,t) -> 0.0)
ch = ConstraintHandler(dh)
add!(ch, dbc)
add!(ch, dbc_outlet)
left_boundary = getfacetset(dh.grid, "left")
close!(ch)
update!(ch, 0.0)

# Mass and Stokes matrix assembly
function assemble_mass_matrix(cellvalues_v::CellValues, cellvalues_p::CellValues, M::SparseMatrixCSC, dh::DofHandler)
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Mₑ = BlockedArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])
    mass_assembler = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(Mₑ, 0)
        Ferrite.reinit!(cellvalues_v, cell)
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
        assemble!(mass_assembler, celldofs(cell), Mₑ)
    end
    return M
end

function assemble_stokes_matrix(cellvalues_v::CellValues, cellvalues_p::CellValues, ν, K::SparseMatrixCSC, dh::DofHandler)
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Kₑ = BlockedArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])
    stiffness_assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Kₑ, 0)
        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_p, cell)
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            for i in 1:n_basefuncs_v
                ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    ∇φⱼ = shape_gradient(cellvalues_v, q_point, j)
                    Kₑ[BlockIndex((v▄, v▄), (i, j))] -= ν * ∇φᵢ ⊡ ∇φⱼ * dΩ
                end
            end
            for j in 1:n_basefuncs_p
                ψ = shape_value(cellvalues_p, q_point, j)
                for i in 1:n_basefuncs_v
                    divφ = shape_divergence(cellvalues_v, q_point, i)
                    Kₑ[BlockIndex((v▄, p▄), (i, j))] += (divφ * ψ) * dΩ
                    Kₑ[BlockIndex((p▄, v▄), (j, i))] += (ψ * divφ) * dΩ
                end
            end
        end
        assemble!(stiffness_assembler, celldofs(cell), Kₑ)
    end
    return K
end

# Global parameters
const ν = 1.0 / 1000.0 # dynamic viscosity
const T = 5.0 # sim time
Δt₀ = 0.01
Δt_save = 0.1

M = allocate_matrix(dh)
M = assemble_mass_matrix(cvu, cvp, M, dh)
K = allocate_matrix(dh)
f = zeros(ndofs(dh))
K = assemble_stokes_matrix(cvu, cvp, ν, K, dh)
apply!(M, ch)
jac_sparsity = sparse(K)

u0 = zeros(ndofs(dh))
apply!(u0, ch)

# RHS structure definition
struct RHSparams
    K::SparseMatrixCSC
    f::Vector
    ch::ConstraintHandler
    dh::DofHandler
    cvu::CellValues
    fvu::FacetValues
    fvp::FacetValues
    boundary
    u::Vector
end
p = RHSparams(K, f, ch, dh, cvu, fvu, fvp, left_boundary, copy(u0))

# Nonlinear total pressure BC parameters
p0 = 1.0 # Pa total pressure
rho = 2.0 # density
α = 100 # NOTE: 1000 didnt work after 4 iterations as dt was becomming too low
const t_ramp = 1.0
α_of_t(t) = α * min(t/t_ramp, 1.0)

# Nonlinear BC functions
total_pressure_res(u_val, p_val, x, t) = (u_val[1]^2 - (2 * (p0 - p_val) / rho)) 
dres_du(u_val, p_val, x, t) = 2.0 * u_val[1]
dres_dp(u_val, p_val, x, t) = -2.0 / rho

function ferrite_limiter!(u, _, p, t)
    Ferrite.update!(p.ch, t)
    return apply!(u, p.ch)
end

# Nonlinear BC residual assembly (for both u and p)
function assemble_nonlinear_residual!(Re::Vector, u_e::Vector, p_e::Vector, fvu::FacetValues, fvp::FacetValues, facet, t::Float64)
    local_ndofs_u = length(u_e)
    nφ_u = div(local_ndofs_u, 2)
    for q_point in 1:getnquadpoints(fvu)
        x = spatial_coordinate(fvu, q_point, getcoordinates(facet))
        dΓ = getdetJdV(fvu, q_point)
        u_q = function_value(fvu, q_point, u_e)
        p_q = function_value(fvp, q_point, p_e)
        res = total_pressure_res(u_q, p_q, x, t)
        for i in 1:nφ_u
            ϕ_vec = shape_value(fvu, q_point, i)
            ϕx = ϕ_vec[1]
            ix = 2*i - 1
            if !(x[2] <= tol_in || x[2] >= H - tol_in)
                Re[ix] -= α_of_t(t) * res * ϕx * dΓ
            end
        end
    end
    return
end

#TODO: Do Stokes Non-linear BC First to get rid of non-linearity in the ODE and then add the non linearity in the BC

# Nonlinear BC Jacobian assembly (for both u and p)
function assemble_nonlinear_jacobian_uublock!(Je::Matrix, u_e::Vector, fvu::FacetValues, facet, t::Float64)
    local_ndofs_u = length(u_e)
    nφ_u = div(local_ndofs_u, 2)
    for q_point in 1:getnquadpoints(fvu)
        x = spatial_coordinate(fvu, q_point, getcoordinates(facet))
        dΓ = getdetJdV(fvu, q_point)
        u_q = function_value(fvu, q_point, u_e)
        dresu = dres_du(u_q, 0.0, x, t) # p not needed for u-u block
        for i in 1:nφ_u
            ϕ_vec_i = shape_value(fvu, q_point, i)
            ϕx_i = ϕ_vec_i[1]
            ix = 2*i - 1
            for j in 1:nφ_u
                ϕ_vec_j = shape_value(fvu, q_point, j)
                ϕx_j = ϕ_vec_j[1]
                jx = 2*j - 1
                if !(x[2] <= tol_in || x[2] >= H - tol_in)
                    Je[ix, jx] -= α_of_t(t) * dresu * ϕx_i * ϕx_j * dΓ
                end
            end
        end
    end
    return
end

# Residual and Jacobian wrappers
function stokes_residual!(R, u_current, p::RHSparams, t::Float64)
    @unpack K, f, ch, dh, cvu, fvu, fvp, boundary, u = p
    u .= u_current
    Ferrite.update!(ch, t)
    apply!(u, ch)
    R .= f
    mul!(R, K, u, -1.0, 1.0)
    u_range = dof_range(dh, :u)
    p_range = dof_range(dh, :p)
    for facet in FacetIterator(dh, boundary)
        Ferrite.reinit!(fvu, facet)
        Ferrite.reinit!(fvp, facet)
        u_boundary_facetdofs = @view celldofs(facet)[u_range]
        p_boundary_facetdofs = @view celldofs(facet)[p_range]
        u_e = similar(u, length(u_boundary_facetdofs)); u_e .= @views u[u_boundary_facetdofs]
        p_e = similar(u, length(p_boundary_facetdofs)); p_e .= @views u[p_boundary_facetdofs]
        Re = zeros(length(u_boundary_facetdofs))
        assemble_nonlinear_residual!(Re, u_e, p_e, fvu, fvp, facet, t)
        assemble!(R, u_boundary_facetdofs, Re)
    end
    return
end

function stokes_jac!(J, u_current, p::RHSparams, t::Float64)
    @unpack K, f, ch, dh, cvu, fvu, fvp, boundary, u = p
    u .= u_current
    Ferrite.update!(ch, t)
    apply!(u, ch)
    nonzeros(J) .= -nonzeros(K)
    assembler = start_assemble(J; fillzero = false)
    u_range = dof_range(dh, :u)
    p_range = dof_range(dh, :p)
    for facet in FacetIterator(dh, boundary)
        Ferrite.reinit!(fvu, facet)
        Ferrite.reinit!(fvp, facet)
        u_boundary_facetdofs = @view celldofs(facet)[u_range]
        p_boundary_facetdofs = @view celldofs(facet)[p_range]
        u_e = similar(u, length(u_boundary_facetdofs)); u_e .= @views u[u_boundary_facetdofs]
        p_e = similar(u, length(p_boundary_facetdofs)); p_e .= @views u[p_boundary_facetdofs]
        Je = zeros(length(u_boundary_facetdofs), length(u_boundary_facetdofs))
        assemble_nonlinear_jacobian_uublock!(Je, u_e, fvu, facet, t)
        assemble!(assembler, u_boundary_facetdofs, Je)
        # Assemble u-p block directly into global J
        nφ_u = div(length(u_e), 2)
        nφ_p = length(p_e)
        for q_point in 1:getnquadpoints(fvu)
            x = spatial_coordinate(fvu, q_point, getcoordinates(facet))
            dΓ = getdetJdV(fvu, q_point)
            u_q = function_value(fvu, q_point, u_e)
            p_q = function_value(fvp, q_point, p_e)
            dresp = dres_dp(u_q, p_q, x, t)
            for i in 1:nφ_u
                ϕ_vec_i = shape_value(fvu, q_point, i)
                ϕx_i = ϕ_vec_i[1]
                ix = 2*i - 1
                global_i = u_boundary_facetdofs[ix]
                for j in 1:nφ_p
                    ψ_j = shape_value(fvp, q_point, j)
                    global_j = p_boundary_facetdofs[j]
                    if !(x[2] <= tol_in || x[2] >= H - tol_in)
                        J[global_i, global_j] -= α_of_t(t) * dresp * ϕx_i * ψ_j * dΓ
                    end
                end
            end
        end
    end
    return apply!(J, ch)
end

rhs = ODEFunction(stokes_residual!, mass_matrix = M; jac = stokes_jac!, jac_prototype = jac_sparsity)
problem = ODEProblem(rhs, u0, (0.0, T), p)

struct FreeDofErrorNorm
    ch::ConstraintHandler
end
(fe_norm::FreeDofErrorNorm)(u::Union{AbstractFloat, Complex}, t) = DiffEqBase.ODE_DEFAULT_NORM(u, t)
(fe_norm::FreeDofErrorNorm)(u::AbstractArray, t) = DiffEqBase.ODE_DEFAULT_NORM(u[fe_norm.ch.free_dofs], t)

timestepper = Rodas5P(autodiff = false, step_limiter! = ferrite_limiter!)

pvd = paraview_collection("stokes-transient-2D-p0-inlet")

# Create output directory if it doesn't exist
import Base.Filesystem: mkpath
mkpath("sol_temp")

# Use integrator/init/intervals pattern as in the tutorial
integrator = init(
    problem, timestepper;
    initializealg = NoInit(),
    dt = Δt₀,
    adaptive = true,
    abstol = 1e-2, reltol = 1e-2,
    progress = true, progress_steps = 1,
    verbose = true, internalnorm = FreeDofErrorNorm(ch), d_discontinuities = [1.0]
)

pvd = paraview_collection("sol_temp/stokes-transient-2D-p0-inlet")
for (step, (u, t)) in enumerate(intervals(integrator))
    VTKGridFile("sol_temp/stokes-transient-2D-p0-inlet$(lpad(step, 4, '0'))", dh) do vtk
        write_solution(vtk, dh, u)
        pvd[t] = vtk
    end
end
vtk_save(pvd)
println("C'est fini")
println("simulation is over")
