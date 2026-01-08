using ReadVTK
using LinearAlgebra

function global_energy(vtkfile; fields=("v","p"))
    vtk = VTKFile(vtkfile)
    pd  = get_point_data(vtk)

    S = 0.0

    if "v" in fields
        v = get_data(pd["v"])
        S += sum(abs2, v)
    end

    if "p" in fields
        p = get_data(pd["p"])
        S += sum(abs2, p)
    end

    return S
end

S_test = global_energy("sol/sol_incomp_w_temp_finished/vortex-street-with-temp-100.vtu")
S_ref  = global_energy("sol/sol_tutorial/vortex-street-100.vtu")

@assert isapprox(S_test, S_ref; rtol=1e-4, atol=1e-4) "Cooked"
