println("Starting execution...")
using Ferrite, SparseArrays, LinearAlgebra, UnPack, LinearSolve, WriteVTK
using OrdinaryDiffEq, DiffEqBase
using FerriteGmsh
using FerriteGmsh: Gmsh
println("Dependencies loaded!")


# Meshing and geometery
Gmsh.initialize()
gmsh.option.set_number("General.Verbosity", 2)
dim = 2
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
ipg = Lagrange{RefQuadrilateral, 1}() # linear geometric interpolation (remi)

# Cell values
qr = QuadratureRule{RefQuadrilateral}(4)
cvu = CellValues(qr, ipu, ipg)
cvp = CellValues(qr, ipp, ipg)

# Facet values used to enforce nl bc on boundary facets
qr_facet = FacetQuadratureRule{RefQuadrilateral}(4)
fvu = FacetValues(qr_facet, ipu, ipg)
fvp = FacetValues(qr_facet, ipp, ipg)

# Dof handler : velocity ux uy and pressure p 
dh = DofHandler(grid)
add!(dh, :u, ipu)
add!(dh, :p, ipp)
close!(dh)


# Boundary conditions easy
ch = ConstraintHandler(dh)

# Walls dirichlet bc on velocity
noslip_boundaries = union(getfacetset(dh.grid, "top"), getfacetset(dh.grid, "bottom"), getfacetset(dh.grid, "hole"))
dbc = Dirichlet(:u, noslip_boundaries, (x, t) -> [0.0, 0.0], [1,2])
add!(ch, dbc)

# Outlet p_static = 0 dirichlet bc on pressure
dbc_outlet = Dirichlet(:p, getfacetset(dh.grid, "right"), (x,t) -> 0.0)
add!(ch, dbc_outlet)

# Finish easy bc work (still need to define inlet nl bc)
left_boundary = getfacetset(dh.grid, "left")
close!(ch)
update!(ch, 0.0); # not sure the point of this tbh


# Mass matrix M and linear stokes matrix K assembly
function assemble_mass_matrix(cellvalues_v::CellValues, cellvalues_p::CellValues, M::SparseMatrixCSC, dh::DofHandler)
    # Allocate a buffer for the local matrix and some helpers, together with the assembler.
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Mₑ = BlockedArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    # It follows the assembly loop as explained in the basic tutorials.
    mass_assembler = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(Mₑ, 0)
        Ferrite.reinit!(cellvalues_v, cell)

        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            # Remember that we assemble a vector mass term, hence the dot product.
            # There is only one time derivative on the left hand side, so only one mass block is non-zero.
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
end;

# NOTE: For now I dont add Remi's fe stuff to this function
function assemble_stokes_matrix(cellvalues_v::CellValues, cellvalues_p::CellValues, ν, K::SparseMatrixCSC, dh::DofHandler)
    # Again, some buffers and helpers
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Kₑ = BlockedArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    # Assembly loop
    stiffness_assembler = start_assemble(K)
    for cell in CellIterator(dh)
        # Don't forget to initialize everything
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

        # Assemble `Kₑ` into the Stokes matrix `K`.
        assemble!(stiffness_assembler, celldofs(cell), Kₑ)
    end
    return K
end;


# Start point ->

# Global parameters
const ν = 1.0 / 1000.0 # dynamic viscocity
const T = 5.0 # sim time
Δt₀ = 0.001
Δt_save = 0.1

# Matrix allocations
M = allocate_matrix(dh);
M = assemble_mass_matrix(cvu, cvp, M, dh);
K = allocate_matrix(dh);
f = zeros(ndofs(dh));
K = assemble_stokes_matrix(cvu, cvp, K, dh);
apply!(M, ch)
jac_sparsity = sparse(K); # might need to change this to different sparse formate if T var is added

u0 = zeros(ndofs(dh));
apply!(u0, ch);


# RHS structure definition 
struct RHSparams
    K::SparseMatrixCSC
    f::Vector
    ch::ConstraintHandler
    dh::DofHandler
    cvu::CellValues
    fvu::FacetValues 
    boundary
    u::Vector
end 
p = RHSparams(K, f, ch, dh, cvu, fvu, left_boundary, copy(u0));

p0 = 1.0 # Pa total pressure
rho = 2.0 # density
# Now, handle the nonlinear BC 
function total_pressure_based_velocity(p_val, x, t) # NOTE: change this to enforce bernoulis equation 
    # y = x[2]
    # H = 1.0
    # ξ = (2*y - H)/H
    # Vmax(t::Float64) = min(t * 2.0, 2.0)
    # vx = Vmax(t) * (1 - ξ^2)
    # return p0 - 0.5*rho*(vx^2)
    vx_expected = sqrt((p0 - p_val) * 2 / rho) # Based on bernoulis equation, p0 = p + 0.5 * rho * v^2
    return vx_expected**2
end;
res_value(u_val, p_val, x, t) = (u_val[1]^2 - total_pressure_based_velocity(p_val, x, t))
dres_value_du(u_val, x) = 2.0 * u_val[1]
dres_value_dp(p_val, x) = 0.5 * (2.0 * p_val / rho)**(-0.5) # TODO: Check

# Defining the penalty with a ramping 
α = 1000
t_ramp = 1.0 # secondes
α_of_t(t) = α * min(t/t_ramp, 1.0)
# 
tol_in = 0.0011
H = 1.0

function ferrite_limiter!(u, _, p, t)
    Ferrite.update!(p.ch, t)
    return apply!(u, p.ch)
end;

# Assembly of the non linear contribution in the residual
function assemble_nonlinear_residual!(Re::Vector, u_e::Vector, p_e::Vector, fvu::FacetValues, cvu::CellValues, facet, t::Float64) # NOTE: I added the p_e::Vector arugment
    local_ndofs_u = length(u_e) # TODO: Idk if local_ndof_u will be same as local_ndof_p
    local_ndof_p = length(p_e) 

    if local_ndof_p == local_ndofs_u:
        println("Warning: local_ndof_p is equal to local_ndof_u, check if this is expected")
    end

    nφ_u = div(local_ndofs_u, 2) 
    n_basefuncs_facet_u = getnbasefunctions(fvu)
    ndofs_u = 2  # nombre de composantes de vitesse en 2D
    # Element residual for the non linear CL
    # Loop over the quadrature points of the facet
    for q_point in 1:getnquadpoints(fvu)
        x = spatial_coordinate(fvu, q_point,getcoordinates(facet))
        dΓ = getdetJdV(fvu, q_point) # getting the weight
        u_q_point = function_value(fvu, q_point, u_e)
        p_q_point = function_value(fvp, q_point, p_e)
        res_q_point = res_value(u_q_point, p_q_point, x, t) # TODO: Ig I need loop over both u and p in the same loop since I need both u_val and p_val to find residual??
        # Loop over the shape functions of the facet 
        for i in 1:nφ_u
            ϕ_vec = shape_value(fvu, q_point, i)  # VectorValue(ϕx, ϕy)
            ϕx = ϕ_vec[1]
            ix = 2*i - 1
            if !(x[2] <= tol_in || x[2] >= H - tol_in) # TODO: Understand whats happining here 
                Re[ix] -= α_of_t(t) * res_q_point * ϕx * dΓ # TODO: Understand whats happening here
            end
        end
    end 
    
    # TODO: Finish/ change if both u and p need to looped once all together.
    local_ndof_p =  lenght(p_e)
    nφ_p = div(local_ndofs_p, 1) # TODO: Idk if this should be 1 or 2 
    n_basefuncs_facet_p = getnbasefunctions(fvp)
    ndofs_p = 1 # Scalar thats why 1 maybe?
    for q_point in 1:getnquadpoints(fvp)
        x = spatial_coordinate(fvp, q_point,getcoordinates(facet))
        dΓ = getdetJdV(fvp, q_point) # getting the weight 
        p_q_point = function_value(fvp, q_point, p_e)
        res_q_point = res_value(p_q_point, x, t) # residual compared to the target value 
        # Loop over the shape functions of the facet 
        for i in 1:nφ_u
            ϕ_vec = shape_value(fvu, q_point, i)  # VectorValue(ϕx, ϕy)
            ϕx = ϕ_vec[1]
            ix = 2*i - 1
            if !(x[2] <= tol_in || x[2] >= H - tol_in) # TODO: Understand whats happining here 
                Re[ix] -= α_of_t(t) * res_q_point * ϕx * dΓ # TODO: Understand whats happening here
            end
        end
    end 
   
    return
end 


