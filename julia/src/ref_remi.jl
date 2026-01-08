using Ferrite
using SparseArrays
using LinearAlgebra         # provides mul! 
using UnPack                # added for time integration using DifferentialEquations.jl 
using OrdinaryDiffEq        # added for time integration using DifferentialEquations.jl
using DiffEqBase # added for time integration using DifferentialEquations.jl   
using WriteVTK

# Mesh definition 
Lx = 4.0
Ly = 1.0 
h_x = 0.04
h_y = 0.001
nx = Int(Lx / h_x ) # 80 / 0.5
ny = Int(Ly / h_y) # 20 / 0.5
## grid creation
grid = generate_grid(Quadrilateral, (nx, ny), Vec((0.0, 0.0)), Vec((Lx, Ly)))
println("Nombre de noeuds : ", length(grid.nodes))
println("Nombre d'éléments : ", length(grid.cells))

# interpolation and quadrature
ipu = Lagrange{RefQuadrilateral, 2}()^2 # quadratic
ipp = Lagrange{RefQuadrilateral, 1}()   # linear
ipg = Lagrange{RefQuadrilateral, 1}() # linear geometric interpolation
qr = QuadratureRule{RefQuadrilateral}(4)
cvu = CellValues(qr, ipu, ipg)
cvp = CellValues(qr, ipp, ipg)
qr_facet = FacetQuadratureRule{RefQuadrilateral}(4)
fvu = FacetValues(qr_facet, ipu, ipg)
fvp = FacetValues(qr_facet, ipp, ipg)

# dof handler 
dh = DofHandler(grid)
add!(dh, :u, ipu)
add!(dh, :p, ipp)
close!(dh)

# constraint handler 
ch = ConstraintHandler(dh)
left_boundary = getfacetset(grid, "left")
## conditions on the top and on the bottom
Γ23 =  union(getfacetset(dh.grid, "top"), getfacetset(dh.grid, "bottom"),)
dbc = Dirichlet(:u, Γ23, (x, t) -> [0.0, 0.0], [1, 2])
add!(ch, dbc)
dbc_outlet = Dirichlet(:p, getfacetset(dh.grid, "right"), (x, t) -> 0.0)
add!(ch,dbc_outlet)
# Finalize
close!(ch)
Ferrite.update!(ch, 0.0)

function assemble_mass_matrix(cellvalues_v::CellValues, cellvalues_p::CellValues, M::SparseMatrixCSC, dh::DofHandler)
    # Allocate a buffer for the local matrix and some helpers, together with the assembler.
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v_block, p_block = 1, 2
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
                    Mₑ[BlockIndex((v_block, v_block), (i, j))] += φᵢ ⋅ φⱼ * dΩ
                end
            end
        end
        assemble!(mass_assembler, celldofs(cell), Mₑ)
    end

    return M
end;




function assemble_stifness_matrix!(K, f, dh, cvu, cvp, μ)
    assembler = start_assemble(K, f)
    n_basefuncs_u = getnbasefunctions(cvu)
    n_basefuncs_p = getnbasefunctions(cvp)
    n_basefuncs = n_basefuncs_u + n_basefuncs_p
    u_block, p_block = 1, 2
    Ke = BlockedArray(zeros(n_basefuncs, n_basefuncs),
                      [n_basefuncs_u, n_basefuncs_p],
                      [n_basefuncs_u, n_basefuncs_p])
    fe = zeros(n_basefuncs)
    # buffers pour fonctions de forme
    ϕᵤ = Vector{Vec{2, Float64}}(undef, n_basefuncs_u)
    ∇ϕᵤ = Vector{Tensor{2, 2, Float64, 4}}(undef, n_basefuncs_u)
    divϕᵤ = Vector{Float64}(undef, n_basefuncs_u)
    ϕₚ = Vector{Float64}(undef, n_basefuncs_p)

    for cell in CellIterator(dh)
        fill!(Ke, 0.0)
        fill!(fe, 0.0)
        Ferrite.reinit!(cvu, cell)
        Ferrite.reinit!(cvp, cell)
        for qp in 1:getnquadpoints(cvu)
            dΩ = getdetJdV(cvu, qp)
            for i in 1:n_basefuncs_u
                ϕᵤ[i] = shape_value(cvu, qp, i)
                ∇ϕᵤ[i] = shape_gradient(cvu, qp, i)
                divϕᵤ[i] = shape_divergence(cvu, qp, i)
            end
            for i in 1:n_basefuncs_p
                ϕₚ[i] = shape_value(cvp, qp, i)
            end
            # bloc u–u
            for i in 1:n_basefuncs_u, j in 1:n_basefuncs_u
                Ke[BlockIndex((u_block, u_block), (i, j))] += μ * (∇ϕᵤ[i] ⊡ ∇ϕᵤ[j]) * dΩ
            end
            # bloc u–p
            for i in 1:n_basefuncs_u, j in 1:n_basefuncs_p
                Ke[BlockIndex((u_block, p_block), (i, j))] += (-divϕᵤ[i] * ϕₚ[j]) * dΩ
            end
            # bloc p–u
            for i in 1:n_basefuncs_p, j in 1:n_basefuncs_u
                Ke[BlockIndex((p_block, u_block), (i, j))] += (-divϕᵤ[j] * ϕₚ[i]) * dΩ
            end

            # second membre (ici nul)
            for i in 1:n_basefuncs_u
                fe[i] += 0.0
            end
        end

        assemble!(assembler, celldofs(cell), Ke, fe)
    end

    return K, f
end;

# dynamic viscosity 
μ = 1
# time
T = 5.0
Δt₀ = 1e-3

# definition of the matrix
M = allocate_matrix(dh);
M = assemble_mass_matrix(cvu, cvp, M, dh);
K = allocate_matrix(dh);
f = zeros(ndofs(dh))
K , f = assemble_stifness_matrix!(K, f, dh, cvu, cvp, μ); # assemble the system
apply!(M,ch)
jac_sparsity = sparse(K);
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
u0 = zeros(ndofs(dh));
# u0 .=1e-9;
apply!(u0, ch);

# Now, handle the nonlinear BC 
## Parabole velocity at the inlet (left boundary)
function parabole_velocity(x,t)
    y = x[2]
    H = 1.0
    ξ = (2*y - H)/H
    Vmax(t::Float64) = min(t * 2.0, 2.0)
    vx = Vmax(t) * (1 - ξ^2)
    vx = (vx^2)
    return vx
end
res_value(u_val, x,t) = (u_val[1]^2- parabole_velocity(x,t)) # à modifier 
dures_value(u_val,x) = 2.0 * u_val[1]

# defining the penalty with a ramping 
α = 1000
t_ramp = 1.0 # secondes
α_of_t(t) = α * min(t/t_ramp, 1.0)
# 
tol_in = 0.0011
H = 1.0

p = RHSparams(K, f, ch, dh, cvu, fvu, left_boundary, copy(u0));

function ferrite_limiter!(u, _, p, t)
    Ferrite.update!(p.ch, t)
    return apply!(u, p.ch)
end;

# Assembly of the non linear contribution in the residual
function assemble_nonlinear_residual!(Re::Vector, u_e::Vector, fvu::FacetValues, cvu::CellValues, facet, t::Float64)
    local_ndofs = length(u_e)
    nφ = div(local_ndofs, 2) 
    n_basefuncs_facet_u = getnbasefunctions(fvu)
    ndofs_u = 2  # nombre de composantes de vitesse en 2D
    # Element residual for the non linear CL
    # Loop over the quadrature points of the facet
    for q_point in 1:getnquadpoints(fvu)
        x = spatial_coordinate(fvu, q_point,getcoordinates(facet))
        dΓ = getdetJdV(fvu, q_point) # getting the weight 
        u_q_point = function_value(fvu, q_point, u_e)
        res_q_point = res_value(u_q_point, x, t) # residual compared to the target value 
        # Loop over the shape functions of the facet 
        for i in 1:nφ
            
            ϕ_vec = shape_value(fvu, q_point, i)  # VectorValue(ϕx, ϕy)
            ϕx = ϕ_vec[1]
            ix = 2*i - 1
            if !(x[2] <= tol_in || x[2] >= H - tol_in)
                Re[ix] -= α_of_t(t) * res_q_point * ϕx * dΓ
            end
            
            #=
            if i%2 ==1 #test to modify only the horizontal velocity component
                #if !(x[2]<= tol_in || x[2] >= H - tol_in)
                ϕ_vec = shape_value(fvu, q_point, i)  # VectorValue(ϕx, ϕy)
                ϕx = ϕ_vec[1]
                Re[i] -= α_of_t(t)  * res_q_point * ϕx * dΓ # contribution to the non linear residual
                #end
            end
            =#
            
        end
    end 
    return
end 

function stokes_residual!(R, u_current, p::RHSparams, t::Float64)
    @unpack K, f, ch, dh, cvu, fvu, boundary, u = p  
    u .= u_current
    Ferrite.update!(ch, t)
    apply!(u, ch)
    ## residual, linear contribution
    R .= f
    mul!(R, K, u, -1.0, 1.0)
    ## residual, non linear contribution 
    u_range = dof_range(dh, :u)
    ndofs_u = 2
    n_basefuncs_facet_u = getnbasefunctions(fvu)
    # Re = zeros(n_basefuncs_facet_u) ############
    #u_e = zeros(n_basefuncs_facet_u) ############
    for facet in FacetIterator(dh, boundary)
        
        Ferrite.reinit!(fvu, facet)
        u_boundary_facetdofs = @view celldofs(facet)[u_range]
        local_ndofs = length(u_boundary_facetdofs)            # = 2 * getnbasefunctions(fvu)
        u_e = similar(u, local_ndofs); u_e .= @views u[u_boundary_facetdofs]
        Re  = zeros(local_ndofs)
    
        assemble_nonlinear_residual!(Re, u_e, fvu, cvu, facet, t)  # ← passe bien t
        assemble!(R, u_boundary_facetdofs, Re)
        #=
        Ferrite.reinit!(fvu, facet)
        u_boundary_facetdofs = @view celldofs(facet)[u_range]
        u_e .= @views u[u_boundary_facetdofs]
        #println("un tour de res")
        #println("u_boundary_facetdofs length: ", length(u_boundary_facetdofs))
        #println("u_e length: ", length(u_e))
        fill!(Re, 0.0)
        assemble_nonlinear_residual!(Re, u_e, fvu, cvu, facet,t)
        assemble!(R, u_boundary_facetdofs, Re)
        #@show u_boundary_facetdofs
        =#
        
        
    end 
    #R[1] = 0 
    #@show R
    #@show u 
    
    return 
end;

function assemble_nonlinear_jac!(Je, u_e::Vector, fvu::FacetValues, cvu::CellValues, facet, t::Float64)
    n_basefuncs_facet_u = getnbasefunctions(fvu)
    local_ndofs = length(u_e)
    nφ = div(local_ndofs, 2) 
    for q_point in 1:getnquadpoints(fvu)
        dΓ = getdetJdV(fvu, q_point) # getting the weight 
        u_q_point = function_value(fvu, q_point, u_e)
        # compute the value that we will be using in the non linear jacobian 
        x = spatial_coordinate(fvu, q_point, getcoordinates(facet))
        du_res_q_point = dures_value(u_q_point,x)  
        # Loop over the test functions of the facet 
        for i in 1:nφ
            
            ϕ_vec = shape_value(fvu, q_point, i)  # VectorValue(ϕx, ϕy)
            ϕ_i_x = ϕ_vec[1]
            ix = 2*i - 1
            for j in 1:nφ 
                ϕ_j = shape_value(fvu, q_point, j)
                ϕ_j_x = ϕ_j[1]
                jx = 2*j - 1
                if !(x[2] <= tol_in || x[2] >= H - tol_in)
                    Je[ix,jx] -= α_of_t(t) * du_res_q_point * ϕ_i_x * ϕ_j_x * dΓ # contribution to the non linear jacobian
                end
            end
            #=
            if i%2 == 1
                ϕ_i = shape_value(fvu, q_point, i)
                ϕ_i_x = ϕ_i[1]
                for j in 1:n_basefuncs_facet_u  # Loop over the trial functions of the facet
                    if j%2==1
                        #if !(x[2]<= tol_in || x[2] >= H - tol_in)
                        ϕ_j = shape_value(fvu, q_point, j)
                        ϕ_j_x = ϕ_j[1]
                        Je[i,j] -= α_of_t(t) * du_res_q_point * ϕ_i_x * ϕ_j_x * dΓ # contribution to the non linear jacobian
                        
                        #end
                    end
                end
            end
            =#
            
        end
    end 
    return
end 

function stokes_jac!(J, u_current, p::RHSparams, t::Float64)
    @unpack  K, f, ch, dh, cvu, fvu, boundary, u = p  # getting the parameters values 
    u .= u_current
    Ferrite.update!(ch, t)
    apply!(u, ch)
    # Linear contribution
    nonzeros(J) .= - nonzeros(K)
    assembler = start_assemble(J; fillzero = false)
    # non linear
    u_range = dof_range(dh, :u)
    ndofs_u = 2
    n_basefuncs_facet_u = getnbasefunctions(fvu)
    #Je = zeros(n_basefuncs_facet_u,n_basefuncs_facet_u) ########
    #u_e = zeros(n_basefuncs_facet_u) #########
    # Non linear contribution
    for facet in FacetIterator(dh, boundary)
        
        Ferrite.reinit!(fvu, facet)
        u_boundary_facetdofs = @view celldofs(facet)[u_range]
        local_ndofs = length(u_boundary_facetdofs)            # = 2 * getnbasefunctions(fvu)
        u_e = similar(u, local_ndofs); u_e .= @views u[u_boundary_facetdofs]
        Je  = zeros(local_ndofs,local_ndofs)
    
        assemble_nonlinear_jac!(Je, u_e, fvu, cvu, facet, t)  
        assemble!(assembler, u_boundary_facetdofs, Je)
        
        #=
        Ferrite.reinit!(fvu, facet)
        u_boundary_facetdofs = @view celldofs(facet)[u_range]
        u_e .= @views u[u_boundary_facetdofs]
        fill!(Je,0.0)
        assemble_nonlinear_jac!(Je, u_e, fvu, cvu, facet, t)
        assemble!(assembler, u_boundary_facetdofs, Je)
        # println("un tour de jac")
        =#

    end
    # println("un tour de jac")
    #@show J-K
    return apply!(J,ch)
end;

#=
εp = 1e-9
M_mod = copy(M)
for j in dof_range(dh, :p)
    M_mod[j,j] += εp
end
=#
rhs = ODEFunction(stokes_residual!, mass_matrix = M; jac = stokes_jac!, jac_prototype = jac_sparsity)
problem = ODEProblem(rhs, u0, (0.0, T), p);

struct FreeDofErrorNorm
    ch::ConstraintHandler
end
(fe_norm::FreeDofErrorNorm)(u::Union{AbstractFloat, Complex}, t) = DiffEqBase.ODE_DEFAULT_NORM(u, t)
(fe_norm::FreeDofErrorNorm)(u::AbstractArray, t) = DiffEqBase.ODE_DEFAULT_NORM(u[fe_norm.ch.free_dofs], t)

timestepper = Rodas5P(autodiff = false, step_limiter! = ferrite_limiter!);

Δt_save = 0.20
sol_stokes = DifferentialEquations.solve(problem, timestepper;
    initializealg = NoInit(),
    dt = Δt₀,
    dtmin = 1e-12,
    adaptive = true,
    abstol = 1e-2, reltol = 1e-2,
    progress = true,
    verbose = true,
    save_start = true,
    save_end = true,
    save_everystep = false,
    saveat = 0:Δt_save:T,
    internalnorm = FreeDofErrorNorm(ch),
    d_discontinuities = [1.0]
);


@show sol_stokes.retcode
@show length(sol_stokes.t), first(sol_stokes.t), last(sol_stokes.t)


pvd = paraview_collection("stokes-transient-2D")
for (k, (t, u)) in enumerate(zip(sol_stokes.t, sol_stokes.u))
    VTKGridFile("stokes-transient-2D$(lpad(k, 4, '0'))", dh) do vtk
        write_solution(vtk, dh, u)                       
        pvd[t]                    = vtk
    end
end
vtk_save(pvd)
println("C'est fini")

println("simulation is over")
