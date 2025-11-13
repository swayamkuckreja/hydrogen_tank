using Ferrite, FerriteGmsh, LinearAlgebra, SparseArrays
using OrdinaryDiffEq, WriteVTK

# --- Physical parameters ---
κ = 1e-3        # diffusion coefficient
v_const = Vec((1.0, 0.0))   # constant velocity field
Tfinal = 2.0
Δt₀ = 0.001

# --- Mesh (same as tutorial) ---
using FerriteGmsh
using FerriteGmsh: Gmsh
Gmsh.initialize()
gmsh.option.set_number("General.Verbosity", 2)
dim = 2;

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
Gmsh.finalize();

# --- FE setup ---
ip_T = Lagrange{RefQuadrilateral, 1}()
qr = QuadratureRule{RefQuadrilateral}(2)
cellvalues_T = CellValues(qr, ip_T)

dh = DofHandler(grid)
add!(dh, :T, ip_T)
close!(dh)

# --- Boundary conditions ---
ch = ConstraintHandler(dh)

# Dirichlet: left = 1, right = 0
∂Ω_left = getfacetset(grid, "left")
∂Ω_right = getfacetset(grid, "right")
add!(ch, Dirichlet(:T, ∂Ω_left, (x, t)->1.0))
add!(ch, Dirichlet(:T, ∂Ω_right, (x, t)->0.0))
close!(ch)
update!(ch, 0.0)

# --- Assembly functions ---
function assemble_mass_matrix(cellvalues_T, M, dh)
    nbf = getnbasefunctions(cellvalues_T)
    Mₑ = zeros(nbf, nbf)
    mass_assembler = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(Mₑ, 0)
        Ferrite.reinit!(cellvalues_T, cell)
        for q in 1:getnquadpoints(cellvalues_T)
            dΩ = getdetJdV(cellvalues_T, q)
            for i in 1:nbf, j in 1:nbf
                ϕi = shape_value(cellvalues_T, q, i)
                ϕj = shape_value(cellvalues_T, q, j)
                Mₑ[i,j] += ϕi * ϕj * dΩ
            end
        end
        assemble!(mass_assembler, celldofs(cell), Mₑ)
    end
    return M
end

function assemble_diffusion_matrix(cellvalues_T, κ, K, dh)
    nbf = getnbasefunctions(cellvalues_T)
    Kₑ = zeros(nbf, nbf)
    stiff_assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Kₑ, 0)
        Ferrite.reinit!(cellvalues_T, cell)
        for q in 1:getnquadpoints(cellvalues_T)
            dΩ = getdetJdV(cellvalues_T, q)
            for i in 1:nbf, j in 1:nbf
                ∇ϕi = shape_gradient(cellvalues_T, q, i)
                ∇ϕj = shape_gradient(cellvalues_T, q, j)
                Kₑ[i,j] += κ * (∇ϕi ⋅ ∇ϕj) * dΩ
            end
        end
        assemble!(stiff_assembler, celldofs(cell), Kₑ)
    end
    return K
end

# NOTE: detailed comments below can be applied to the other 2 matrices assembled above also
function assemble_advection_matrix(cellvalues_T, v, A, dh)
    nbf = getnbasefunctions(cellvalues_T) # this is my number of nodes i think
    Aₑ = zeros(nbf, nbf) # initialize the cell matrix A_e
    adv_assembler = start_assemble(A) # needed in conjunction with assemble! to start the global matrix assembly process
    for cell in CellIterator(dh) # loop over all cells 
        fill!(Aₑ, 0) # every cell needs to assemble its own matrix 
        Ferrite.reinit!(cellvalues_T, cell) # need to reload the cell values for each cell 
        for q in 1:getnquadpoints(cellvalues_T) # need to loop over all quad points for numerical integration
            dΩ = getdetJdV(cellvalues_T, q)
            for i in 1:nbf, j in 1:nbf # loop over i and j trial and test function indicies (as derived)
                ϕi = shape_value(cellvalues_T, q, i)
                ∇ϕj = shape_gradient(cellvalues_T, q, j)
                Aₑ[i,j] += ϕi * (v ⋅ ∇ϕj) * dΩ # this is the FEM stuff i derive
            end
        end
        assemble!(adv_assembler, celldofs(cell), Aₑ) # adds the cells contribution to the global matrices
    end
    return A
end

# --- Build matrices ---
M = allocate_matrix(dh)
K = allocate_matrix(dh)
A = allocate_matrix(dh)

M = assemble_mass_matrix(cellvalues_T, M, dh)
K = assemble_diffusion_matrix(cellvalues_T, κ, K, dh)
A = assemble_advection_matrix(cellvalues_T, v_const, A, dh)

# Apply Dirichlet BCs to matrices, TODO: no need to do this here apprently as ODEProblem does this ig?
#apply!(M, ch)
#apply!(K, ch)
#apply!(A, ch)

# --- ODE setup ---
T₀ = zeros(ndofs(dh))
apply!(T₀, ch)

function rhs_T!(dT, T, p, t)
    # Unpack parameters explicitly (avoid needing UnPack.jl)
    M = p.M
    A = p.A
    K = p.K
    ch = p.ch
    apply!(T, ch)
    dT .= -(A + K) * T
end

params = (; M, A, K, ch)

rhs = ODEFunction(rhs_T!, mass_matrix = M)
problem = ODEProblem(rhs, T₀, (0.0, Tfinal), params)

tspan = 0:0.02:Tfinal   # 100 frames over Tfinal=2.0

sol = solve(problem, Rodas5P(), dt=Δt₀, adaptive=true, saveat=tspan)

pvd = paraview_collection("temp_advdiff")
for (i, T) in enumerate(sol.u)
    VTKGridFile("temp-$i", dh) do vtk
        write_solution(vtk, dh, T)
        pvd[sol.t[i]] = vtk
    end
end
vtk_save(pvd)

