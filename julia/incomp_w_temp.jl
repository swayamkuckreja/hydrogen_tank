using Ferrite, SparseArrays, BlockArrays, LinearAlgebra, UnPack, LinearSolve, WriteVTK

using OrdinaryDiffEq
using DiffEqBase

ν = 1.0 / 1000.0; #dynamic viscosity

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

# above code copied from tutorial_incomp.jl ^ #

ip_v = Lagrange{RefQuadrilateral, 2}()^dim # is for quadratic shape functions (TODO: try 3 here)
ip_p = Lagrange{RefQuadrilateral, 1}()
ip_T = Lagrange{RefQuadrilateral, 1}()

# num integration method
qr = QuadratureRule{RefQuadrilateral}(4)

# idk what is for prolly some FEM stuff
cellvalues_v = CellValues(qr, ip_v)
cellvalues_p = CellValues(qr, ip_p)
cellvalues_T = CellValues(qr, ip_T)

# defining variables
dh = DofHandler(grid)
add!(dh, :v, ip_v)
add!(dh, :p, ip_p)
add!(dh, :T, ip_T)
close!(dh);

ch = ConstraintHandler(dh)

# TODO: Figure out the ch syntax later

close!(ch)

#


