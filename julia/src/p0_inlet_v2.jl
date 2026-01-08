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

# Matrix creations
# 
