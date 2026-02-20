println("Starting execution...")
using Ferrite, SparseArrays, LinearAlgebra, UnPack, LinearSolve, WriteVTK, BlockArrays
using OrdinaryDiffEq, DiffEqBase
using FerriteGmsh
using FerriteGmsh: Gmsh
println("Dependencies loaded!")

# =============================================================================
# GOVERNING EQUATIONS
# =============================================================================
# We solve the transient incompressible *Stokes* equations:
#
#   ∂u/∂t = ν Δu - ∇p    (momentum, without the nonlinear (u·∇)u term)
#   ∇·u   = 0             (incompressibility)
#
# Semi-discrete (after FEM in space):
#   M dU/dt = K U + N_BC(U)
#
# where:
#   M   — singular mass matrix (velocity block only; pressure has no ∂/∂t)
#   K   — Stokes stiffness matrix (viscous + pressure/divergence)
#   N_BC — nonlinear *boundary* contribution from the p0 inlet condition
#
# INLET BOUNDARY CONDITION — Total Pressure (p0) condition
# ---------------------------------------------------------
# Instead of prescribing velocity at the inlet, we prescribe the *total* pressure:
#   p0 = p_static + ½ρ u_x²     (Bernoulli along a streamline)
#
# Rearranged to a residual that must vanish at the inlet:
#   r(u, p) = u_x² - 2(p0 - p)/ρ = 0
#
# This is enforced via a penalty method. The contribution to the RHS is:
#   F_i = -α(t) · r(u_x, p) · φ_x_i  dΓ        (drives r → 0)
#
# Jacobian blocks:
#   ∂F_i/∂u_x_j = -α · 2 u_x · φ_x_i · φ_x_j  dΓ    [u-u block]
#   ∂F_i/∂p_j   = -α · (2/ρ) · φ_x_i · ψ_j    dΓ    [u-p block]
#
# KEY DIFFERENCES FROM p0_inlet_v3.jl:
#   - No (u·∇)u convection term → purely linear PDE, nonlinearity only in BC
#   - Cleaner sign convention following incomp_w_temp.jl (R = +K*u + N_BC)
#   - Explicit inline Jacobian u-p block in the dedicated Jacobian function
# =============================================================================

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
dim = 2
Gmsh.initialize()
gmsh.option.set_number("General.Verbosity", 2)

rect_tag = gmsh.model.occ.add_rectangle(0, 0, 0, 1.1, 0.41)
circle_tag = gmsh.model.occ.add_circle(0.2, 0.2, 0, 0.05)
circle_curve_tag = gmsh.model.occ.add_curve_loop([circle_tag])
circle_surf_tag = gmsh.model.occ.add_plane_surface([circle_curve_tag])
gmsh.model.occ.cut([(dim, rect_tag)], [(dim, circle_surf_tag)])
gmsh.model.occ.synchronize()

# Physical groups — tag IDs match the default Gmsh ordering after the boolean cut
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
println("Mesh: $(getncells(grid)) cells, $(getnnodes(grid)) nodes")

# ---------------------------------------------------------------------------
# Finite element spaces
# ---------------------------------------------------------------------------
# Taylor-Hood Q2/Q1 pair — satisfies the LBB inf-sup condition
ipu = Lagrange{RefQuadrilateral, 2}()^dim   # Q2 vector for velocity
ipp = Lagrange{RefQuadrilateral, 1}()        # Q1 scalar for pressure
ipg = Lagrange{RefQuadrilateral, 1}()        # Q1 geometric interpolation

# Volume integration
qr = QuadratureRule{RefQuadrilateral}(4)
cvu = CellValues(qr, ipu, ipg)
cvp = CellValues(qr, ipp, ipg)

# Facet (boundary) integration — needed for the p0 nonlinear BC
qr_facet = FacetQuadratureRule{RefQuadrilateral}(4)
fvu = FacetValues(qr_facet, ipu, ipg)
fvp = FacetValues(qr_facet, ipp, ipg)

# DOF handler: velocity u and pressure p
dh = DofHandler(grid)
add!(dh, :u, ipu)
add!(dh, :p, ipp)
close!(dh)

println("Total DOFs: $(ndofs(dh))")

# ---------------------------------------------------------------------------
# Boundary conditions (Dirichlet only — inlet is handled via penalty NL BC)
# ---------------------------------------------------------------------------
ch = ConstraintHandler(dh)

# No-slip walls and cylinder surface
noslip_facets = union(
    getfacetset(grid, "top"),
    getfacetset(grid, "bottom"),
    getfacetset(grid, "hole")
)
add!(ch, Dirichlet(:u, noslip_facets, (x, t) -> Vec((0.0, 0.0)), [1, 2]))

# Outlet: static pressure = 0 (outflow / do-nothing in pressure)
add!(ch, Dirichlet(:p, getfacetset(grid, "right"), (x, t) -> 0.0))

# The LEFT (inlet) boundary has NO Dirichlet BC — driven entirely by the
# penalised p0 condition assembled in N_BC.
left_boundary = getfacetset(grid, "left")

close!(ch)
update!(ch, 0.0)

# Tolerance to exclude corner nodes (where walls and inlet meet)
const tol_corner = 0.0011
const H_channel = 0.41   # channel height [m]

# ---------------------------------------------------------------------------
# Mass matrix assembly
# M only has the velocity-velocity block; pressure has no time derivative.
# ---------------------------------------------------------------------------
function assemble_mass_matrix(cvu::CellValues, cvp::CellValues, M::SparseMatrixCSC, dh::DofHandler)
    nv = getnbasefunctions(cvu)
    np = getnbasefunctions(cvp)
    n  = nv + np
    v▄, p▄ = 1, 2
    Mₑ = BlockedArray(zeros(n, n), [nv, np], [nv, np])
    assembler = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(Mₑ, 0)
        Ferrite.reinit!(cvu, cell)
        for qp in 1:getnquadpoints(cvu)
            dΩ = getdetJdV(cvu, qp)
            for i in 1:nv
                φᵢ = shape_value(cvu, qp, i)
                for j in 1:nv
                    φⱼ = shape_value(cvu, qp, j)
                    Mₑ[BlockIndex((v▄, v▄), (i, j))] += φᵢ ⋅ φⱼ * dΩ
                end
            end
        end
        # Pressure block is zero — no ∂p/∂t in Stokes
        assemble!(assembler, celldofs(cell), Mₑ)
    end
    return M
end

# ---------------------------------------------------------------------------
# Stokes stiffness matrix assembly
#
# K = [ K_vv   K_vp ]     K_vv[i,j] = -ν ∫ ∇φᵢ : ∇φⱼ dΩ
#     [ K_pv   0    ]     K_vp[i,j] = +∫ (∇·φᵢ) ψⱼ dΩ
#                          K_pv      = K_vp^T     (incompressibility constraint)
# ---------------------------------------------------------------------------
function assemble_stokes_matrix(cvu::CellValues, cvp::CellValues, ν, K::SparseMatrixCSC, dh::DofHandler)
    nv = getnbasefunctions(cvu)
    np = getnbasefunctions(cvp)
    n  = nv + np
    v▄, p▄ = 1, 2
    Kₑ = BlockedArray(zeros(n, n), [nv, np], [nv, np])
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Kₑ, 0)
        Ferrite.reinit!(cvu, cell)
        Ferrite.reinit!(cvp, cell)
        for qp in 1:getnquadpoints(cvu)
            dΩ = getdetJdV(cvu, qp)
            # Viscous block: K_vv[i,j] = -ν ∫ ∇φᵢ ⊡ ∇φⱼ dΩ
            for i in 1:nv
                ∇φᵢ = shape_gradient(cvu, qp, i)
                for j in 1:nv
                    ∇φⱼ = shape_gradient(cvu, qp, j)
                    Kₑ[BlockIndex((v▄, v▄), (i, j))] -= ν * ∇φᵢ ⊡ ∇φⱼ * dΩ
                end
            end
            # Pressure-velocity coupling: K_vp and K_pv
            for j in 1:np
                ψⱼ = shape_value(cvp, qp, j)
                for i in 1:nv
                    divφᵢ = shape_divergence(cvu, qp, i)
                    Kₑ[BlockIndex((v▄, p▄), (i, j))] += divφᵢ * ψⱼ * dΩ
                    Kₑ[BlockIndex((p▄, v▄), (j, i))] += ψⱼ * divφᵢ * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Kₑ)
    end
    return K
end

# ---------------------------------------------------------------------------
# Physical / simulation parameters
# ---------------------------------------------------------------------------
const ν     = 1.0 / 1000.0   # kinematic viscosity [m²/s]
const p0    = 1.0             # inlet total pressure [Pa]
const ρ     = 2.0             # fluid density [kg/m³]
const T_end = 5.0             # simulation end time [s]
const Δt₀   = 0.01            # initial time step [s]

# Penalty coefficient for the p0 BC — ramps up over t_ramp to avoid stiff start
const α     = 100.0
const t_ramp = 1.0
α_of_t(t) = α * min(t / t_ramp, 1.0)

# ---------------------------------------------------------------------------
# Allocate and assemble linear matrices
# ---------------------------------------------------------------------------
M = allocate_matrix(dh)
M = assemble_mass_matrix(cvu, cvp, M, dh)
K = allocate_matrix(dh)
K = assemble_stokes_matrix(cvu, cvp, ν, K, dh)
apply!(M, ch)            # modify M rows/cols for constrained DOFs
jac_sparsity = sparse(K) # sparsity pattern for the Jacobian

u0 = zeros(ndofs(dh))
apply!(u0, ch)           # initial condition satisfies Dirichlet BCs

# ---------------------------------------------------------------------------
# Nonlinear BC functions
#
# Residual: r(u_x, p) = u_x² - 2(p0 - p)/ρ
#   = 0 when Bernoulli is satisfied at the inlet
# ---------------------------------------------------------------------------
@inline nl_res(ux, p_val)  = ux^2 - 2 * (p0 - p_val) / ρ
@inline dnl_res_dux(ux)    = 2.0 * ux
@inline dnl_res_dp()       = 2.0 / ρ

# ---------------------------------------------------------------------------
# RHSparams struct — carries all needed objects into the ODE RHS / Jacobian
# ---------------------------------------------------------------------------
struct RHSparams
    K::SparseMatrixCSC
    ch::ConstraintHandler
    dh::DofHandler
    cvu::CellValues
    fvu::FacetValues
    fvp::FacetValues
    boundary    # FacetSet for the inlet boundary
    u::Vector   # scratch space (constraint-applied copy of current solution)
end
p = RHSparams(K, ch, dh, cvu, fvu, fvp, left_boundary, copy(u0))

# Step limiter: called after each accepted step to re-enforce BCs
function ferrite_limiter!(u, _, p, t)
    Ferrite.update!(p.ch, t)
    return apply!(u, p.ch)
end

# ---------------------------------------------------------------------------
# Global ODE RHS:  du/dt = M⁻¹ R(u)  with  R(u) = K u + N_BC(u)
#
# For pure Stokes (no convection):
#   R(u) = K u   [linear everywhere]
#         + sum over inlet facets of: -α r(u_x, p) φ_x_i dΓ
#
# There is NO (u·∇u) term — that is the key simplification vs Navier-Stokes.
# ---------------------------------------------------------------------------
function stokes_rhs!(du, u_uc, p::RHSparams, t)
    @unpack K, ch, dh, cvu, fvu, fvp, boundary, u = p

    # Apply constraints to get a consistent solution vector
    u .= u_uc
    Ferrite.update!(ch, t)
    apply!(u, ch)

    # Linear Stokes contribution:  du = K u
    mul!(du, K, u)

    # Nonlinear inlet BC penalty contribution
    u_range = dof_range(dh, :u)
    p_range = dof_range(dh, :p)

    for facet in FacetIterator(dh, boundary)
        Ferrite.reinit!(fvu, facet)
        Ferrite.reinit!(fvp, facet)

        celld = celldofs(facet)
        u_dofs = @view celld[u_range]
        p_dofs = @view celld[p_range]
        u_e = u[u_dofs]
        p_e = u[p_dofs]

        nφ_u = getnbasefunctions(fvu)  # = 18 for Q2^2 (9 nodes × 2 components)
        Re = zeros(length(u_dofs))        # also length 18

        for qp in 1:getnquadpoints(fvu)
            x  = spatial_coordinate(fvu, qp, getcoordinates(facet))
            dΓ = getdetJdV(fvu, qp)
            ux = function_value(fvu, qp, u_e)[1]      # x-component of velocity
            pq = function_value(fvp, qp, p_e)         # pressure at this point
            res = nl_res(ux, pq)

            # Skip corner nodes where inlet meets no-slip walls
            if x[2] <= tol_corner || x[2] >= H_channel - tol_corner
                continue
            end

            for i in 1:nφ_u
                # shape_value returns Vec{2}; [1] is x-component, which is zero
                # for y-DOFs (even i) — so those rows get no contribution naturally
                φ_x = shape_value(fvu, qp, i)[1]
                Re[i] -= α_of_t(t) * res * φ_x * dΓ
            end
        end

        assemble!(du, u_dofs, Re)
    end

    return
end

# ---------------------------------------------------------------------------
# Jacobian:  J = dR/du = K + dN_BC/du
#
# The nonlinear BC Jacobian has two blocks:
#   [u,u]:  ∂(F_i)/∂(u_x_j) = -α · (2 u_x) · φ_x_i · φ_x_j  dΓ
#   [u,p]:  ∂(F_i)/∂(p_j)   = -α · (2/ρ)   · φ_x_i · ψ_j     dΓ
#
# The pressure-velocity and pressure-pressure blocks from K are already in J.
# ---------------------------------------------------------------------------
function stokes_jac!(J::SparseMatrixCSC, u_uc::Vector, p::RHSparams, t::Float64)
    @unpack K, ch, dh, cvu, fvu, fvp, boundary, u = p

    u .= u_uc
    Ferrite.update!(ch, t)
    apply!(u, ch)

    # Start from the linear stiffness
    J .= K

    u_range = dof_range(dh, :u)
    p_range = dof_range(dh, :p)

    assembler = start_assemble(J; fillzero = false)

    for facet in FacetIterator(dh, boundary)
        Ferrite.reinit!(fvu, facet)
        Ferrite.reinit!(fvp, facet)

        celld = celldofs(facet)
        u_dofs = @view celld[u_range]
        p_dofs = @view celld[p_range]
        u_e = u[u_dofs]
        p_e = u[p_dofs]

        nφ_u = getnbasefunctions(fvu)  # = 18 for Q2^2
        nφ_p = getnbasefunctions(fvp)

        # [u,u] block Jacobian (18×18; only x-DOF rows/cols get nonzero entries)
        Juu = zeros(length(u_dofs), length(u_dofs))

        for qp in 1:getnquadpoints(fvu)
            x  = spatial_coordinate(fvu, qp, getcoordinates(facet))
            dΓ = getdetJdV(fvu, qp)
            ux = function_value(fvu, qp, u_e)[1]
            pq = function_value(fvp, qp, p_e)
            d_res_dux = dnl_res_dux(ux)
            d_res_dp  = dnl_res_dp()

            if x[2] <= tol_corner || x[2] >= H_channel - tol_corner
                continue
            end

            for i in 1:nφ_u
                φ_x_i = shape_value(fvu, qp, i)[1]  # zero for y-DOFs, so safe to loop all

                # [u,u] block: ∂F_i/∂u_x_j — index directly, φ_x handles sparsity
                for j in 1:nφ_u
                    φ_x_j = shape_value(fvu, qp, j)[1]
                    Juu[i, j] -= α_of_t(t) * d_res_dux * φ_x_i * φ_x_j * dΓ
                end

                # [u,p] block: ∂F_i/∂p_j — scatter directly into global J
                global_i = u_dofs[i]
                for j in 1:nφ_p
                    ψⱼ = shape_value(fvp, qp, j)
                    global_j = p_dofs[j]
                    J[global_i, global_j] -= α_of_t(t) * d_res_dp * φ_x_i * ψⱼ * dΓ
                end
            end
        end

        assemble!(assembler, u_dofs, Juu)
    end

    # Enforce Dirichlet BCs in Jacobian
    return apply!(J, ch)
end

# ---------------------------------------------------------------------------
# ODE problem setup
# ---------------------------------------------------------------------------
println("Assembling system...")

rhs_fn = ODEFunction(
    stokes_rhs!;
    mass_matrix   = M,
    jac           = stokes_jac!,
    jac_prototype = jac_sparsity
)
problem = ODEProblem(rhs_fn, u0, (0.0, T_end), p)

# Error norm restricted to free DOFs (constrained DOFs must not inflate the norm)
struct FreeDofErrorNorm
    ch::ConstraintHandler
end
(fe_norm::FreeDofErrorNorm)(u::Union{AbstractFloat, Complex}, t) = DiffEqBase.ODE_DEFAULT_NORM(u, t)
(fe_norm::FreeDofErrorNorm)(u::AbstractArray, t) = DiffEqBase.ODE_DEFAULT_NORM(u[fe_norm.ch.free_dofs], t)

# Rodas5P — 5th-order L-stable Rosenbrock method for stiff DAEs
timestepper = Rodas5P(autodiff = false, step_limiter! = ferrite_limiter!)

integrator = init(
    problem, timestepper;
    initializealg  = NoInit(),
    dt             = Δt₀,
    adaptive       = true,
    abstol         = 1e-3,
    reltol         = 1e-3,
    progress       = true, progress_steps = 1,
    verbose        = true,
    internalnorm   = FreeDofErrorNorm(ch),
    d_discontinuities = [t_ramp]   # penalty ramp has a kink here
)

# ---------------------------------------------------------------------------
# Time integration + VTK output
# ---------------------------------------------------------------------------
println("Solving...")
import Base.Filesystem: mkpath
mkpath("sol/sol_stokes_p0")

pvd = paraview_collection("sol/sol_stokes_p0/stokes-p0-inlet")
for (step, (u, t)) in enumerate(intervals(integrator))
    println("Step $step  |  t = $(round(t; digits=4)) s  |  max|u| = $(round(maximum(abs, u); sigdigits=4))")
    VTKGridFile("sol/sol_stokes_p0/stokes-p0-inlet-$(lpad(step, 4, '0'))", dh) do vtk
        write_solution(vtk, dh, u)
        pvd[t] = vtk
    end
end
vtk_save(pvd)
println("Done! Open 'sol/sol_stokes_p0/stokes-p0-inlet.pvd' in Paraview.")
