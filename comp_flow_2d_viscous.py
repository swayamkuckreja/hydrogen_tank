# ns2d_viscous.py
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------------
# Physical constants (air/H2 placeholders)
# ---------------------------
GAMMA = 1.4
R = 287.05               # J/(kg K) (use hydrogen-specific R if modelling H2)
# You can replace R with R_H2 (~4124 J/(kmol*K) / M_molar etc.) but keep units consistent.

# ---------------------------
# Domain and grid
# ---------------------------
Lx = 10.0
Ly = 5.0
Nx = 100
Ny = 100
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# ---------------------------
# Initial & boundary conditions (base)
# ---------------------------
P0 = 101325.0
T0 = 300.0
rho0 = P0 / (R * T0)
u0 = 50.0
v0 = 0.0

# ---------------------------
# Time stepping control
# ---------------------------
CFL_conv = 0.3     # convective CFL
CFL_visc = 0.25    # diffusive CFL
dt_max = 1e-4      # safety cap
nt = 2000

# ---------------------------
# Viscous & thermal properties
# ---------------------------
mu_const = 1.0e-5    # dynamic viscosity [Pa·s] (tune for H2 or air)
Pr = 0.72            # Prandtl number (tune for H2)
cp = GAMMA * R / (GAMMA - 1)
k_const = mu_const * cp / Pr   # thermal conductivity via k = mu*cp/Pr

# ---------------------------
# Conserved variables U = [rho, rho*u, rho*v, E]
# ---------------------------
U = np.zeros((Nx, Ny, 4))
U[:, :, 0] = rho0
U[:, :, 1] = rho0 * u0
U[:, :, 2] = rho0 * v0
U[:, :, 3] = P0 / (GAMMA - 1) + 0.5 * rho0 * (u0**2 + v0**2)

# ---------------------------
# Utility: primitives
# ---------------------------
def primitive_vars(U):
    rho = U[:, :, 0]
    u = U[:, :, 1] / rho
    v = U[:, :, 2] / rho
    E = U[:, :, 3]
    p = (GAMMA - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
    # numerical safety: avoid negative pressure
    p = np.maximum(p, 1e-8)
    T = p / (rho * R)
    return rho, u, v, p, T

# ---------------------------
# Convective fluxes (cell-centered)
# ---------------------------
def flux_x_conv(U):
    rho, u, v, p, _ = primitive_vars(U)
    Fx = np.zeros_like(U)
    Fx[:, :, 0] = rho * u
    Fx[:, :, 1] = rho * u**2 + p
    Fx[:, :, 2] = rho * u * v
    Fx[:, :, 3] = u * (U[:, :, 3] + p)
    return Fx

def flux_y_conv(U):
    rho, u, v, p, _ = primitive_vars(U)
    Fy = np.zeros_like(U)
    Fy[:, :, 0] = rho * v
    Fy[:, :, 1] = rho * u * v
    Fy[:, :, 2] = rho * v**2 + p
    Fy[:, :, 3] = v * (U[:, :, 3] + p)
    return Fy

# ---------------------------
# Viscous fluxes (central diffs)
# ---------------------------
def viscous_flux_x(U, mu=mu_const, k=k_const):
    rho, u, v, p, T = primitive_vars(U)
    Fxv = np.zeros_like(U)

    # gradients (central differences) - zero arrays already
    dudx = np.zeros_like(u); dudy = np.zeros_like(u)
    dvdx = np.zeros_like(v); dvdy = np.zeros_like(v)
    dTdx = np.zeros_like(T)

    dudx[1:-1,1:-1] = (u[2:,1:-1] - u[:-2,1:-1]) / (2*dx)
    dudy[1:-1,1:-1] = (u[1:-1,2:] - u[1:-1,:-2]) / (2*dy)
    dvdx[1:-1,1:-1] = (v[2:,1:-1] - v[:-2,1:-1]) / (2*dx)
    dvdy[1:-1,1:-1] = (v[1:-1,2:] - v[1:-1,:-2]) / (2*dy)
    dTdx[1:-1,1:-1] = (T[2:,1:-1] - T[:-2,1:-1]) / (2*dx)

    lam = -2.0/3.0 * mu

    tau_xx = 2*mu*dudx + lam*(dudx + dvdy)
    tau_xy = mu*(dudy + dvdx)
    qx = -k * dTdx

    Fxv[:,:,0] = 0.0
    Fxv[:,:,1] = tau_xx
    Fxv[:,:,2] = tau_xy
    Fxv[:,:,3] = u * tau_xx + v * tau_xy + qx
    return Fxv

def viscous_flux_y(U, mu=mu_const, k=k_const):
    rho, u, v, p, T = primitive_vars(U)
    Fyv = np.zeros_like(U)

    dudx = np.zeros_like(u); dudy = np.zeros_like(u)
    dvdx = np.zeros_like(v); dvdy = np.zeros_like(v)
    dTdy = np.zeros_like(T)

    dudx[1:-1,1:-1] = (u[2:,1:-1] - u[:-2,1:-1]) / (2*dx)
    dudy[1:-1,1:-1] = (u[1:-1,2:] - u[1:-1,:-2]) / (2*dy)
    dvdx[1:-1,1:-1] = (v[2:,1:-1] - v[:-2,1:-1]) / (2*dx)
    dvdy[1:-1,1:-1] = (v[1:-1,2:] - v[1:-1,:-2]) / (2*dy)
    dTdy[1:-1,1:-1] = (T[1:-1,2:] - T[1:-1,:-2]) / (2*dy)

    lam = -2.0/3.0 * mu

    tau_yy = 2*mu*dvdy + lam*(dudx + dvdy)
    tau_xy = mu*(dudy + dvdx)
    qy = -k * dTdy

    Fyv[:,:,0] = 0.0
    Fyv[:,:,1] = tau_xy
    Fyv[:,:,2] = tau_yy
    Fyv[:,:,3] = u * tau_xy + v * tau_yy + qy
    return Fyv

# ---------------------------
# Boundary conditions for viscous solver
# - Left: inlet (fixed primitive)
# - Right: outlet (zero-gradient)
# - Bottom: no-slip, fixed T (example)
# - Top: no-slip, adiabatic (zero heat flux)
# ---------------------------
def apply_boundary_viscous(U, T_wall_bottom=350.0):
    # Inlet (left): fixed primitive values
    U[0, :, 0] = rho0
    U[0, :, 1] = rho0 * u0
    U[0, :, 2] = rho0 * v0
    U[0, :, 3] = P0 / (GAMMA - 1) + 0.5 * rho0 * (u0**2 + v0**2)

    # Outlet (right): zero-gradient (copy interior)
    U[-1, :, :] = U[-2, :, :]

    # Bottom wall (j=0): no-slip + fixed temperature
    # set velocities to zero (ρu, ρv)
    U[:, 0, 1] = 0.0
    U[:, 0, 2] = 0.0
    # density: copy interior or compute from ideal gas if desired
    rho_b = U[:, 1, 0].copy()
    U[:, 0, 0] = rho_b
    p_b = rho_b * R * T_wall_bottom
    U[:, 0, 3] = p_b / (GAMMA - 1) + 0.5 * rho_b * (0.0**2 + 0.0**2)

    # Top wall (j=Ny-1): no-slip + adiabatic (zero T gradient)
    U[:, -1, 1] = 0.0
    U[:, -1, 2] = 0.0
    # copy density and energy from adjacent interior (zero gradient in T -> copy E)
    U[:, -1, 0] = U[:, -2, 0]
    U[:, -1, 3] = U[:, -2, 3]

    return U

# ---------------------------
# Time-step calculation: convective + diffusive constraints
# ---------------------------
def compute_dt(U):
    rho, u, v, p, T = primitive_vars(U)
    a = np.sqrt(GAMMA * p / rho)  # speed of sound
    max_speed = np.max(np.abs(u) + a)
    if max_speed < 1e-8:
        dt_c = 1e-6
    else:
        dt_c = CFL_conv * min(dx, dy) / max_speed

    # viscous limit: use typical maximum kinematic viscosity
    nu = mu_const / np.maximum(rho, 1e-8)  # elementwise
    # for diffusive dt estimate we use min of dx^2/nu over domain (take max nu -> worst case)
    max_nu = np.max(nu)
    if max_nu <= 0:
        dt_v = dt_max
    else:
        dt_v = CFL_visc * min(dx*dx, dy*dy) / max_nu

    dt = min(dt_c, dt_v, dt_max)
    return dt

# ---------------------------
# Main time-marching routine
# ---------------------------
def main():
    global U
    U = apply_boundary_viscous(U)  # enforce initial BCs
    history = []
    for n in range(nt):
        dt = compute_dt(U)

        Fc = flux_x_conv(U)
        Gc = flux_y_conv(U)
        Fv = viscous_flux_x(U, mu_const, k_const)
        Gv = viscous_flux_y(U, mu_const, k_const)

        U_new = U.copy()

        # interior update (1:-1,1:-1)
        # convective: central difference (second-order)
        conv_term = - dt/(2.0*dx) * (Fc[2:,1:-1] - Fc[:-2,1:-1]) \
                    - dt/(2.0*dy) * (Gc[1:-1,2:] - Gc[1:-1,:-2])

        # viscous: divergence of viscous flux (note sign: + divergence on RHS)
        visc_term = dt/(2.0*dx) * (Fv[2:,1:-1] - Fv[:-2,1:-1]) \
                   + dt/(2.0*dy) * (Gv[1:-1,2:] - Gv[1:-1,:-2])

        # a small smoothing average (like original). Keeps it optional:
        smoothing = 0.25 * (U[2:,1:-1] + U[:-2,1:-1] + U[1:-1,2:] + U[1:-1,:-2])

        U_new[1:-1,1:-1] = smoothing + conv_term + visc_term

        # enforce BCs
        U_new = apply_boundary_viscous(U_new)

        U = U_new

        # monitoring
        rho, u, v, p, T = primitive_vars(U)
        if n % 100 == 0 or n == nt-1:
            print(f"Step {n+1}/{nt} dt={dt:.2e} | mean u={np.mean(u):.3f} | max u={np.max(u):.3f} | max T={np.max(T):.2f}")
        # store occasional fields (optional)
        if n % 200 == 0:
            history.append((n, rho.copy(), u.copy(), p.copy(), T.copy()))

    # final primitives
    rho, u, v, p, T = primitive_vars(U)

    # ---------------------------
    # Plots
    # ---------------------------
    plt.figure(figsize=(12, 8))
    for i, (data, title) in enumerate(zip([rho, u, p, T],
                                         ['Density [kg/m^3]', 'u-velocity [m/s]', 'Pressure [Pa]', 'Temperature [K]'])):
        plt.subplot(2,2,i+1)
        plt.contourf(X, Y, data, 50)
        plt.colorbar()
        plt.title(title)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
    plt.tight_layout()
    plt.show()

    # Streamlines
    plt.figure(figsize=(8, 4))
    speed = np.sqrt(u**2 + v**2)
    lw = 2 * speed / (speed.max() + 1e-12)
    plt.streamplot(X.T, Y.T, u.T, v.T, color=speed.T, linewidth=lw.T, cmap='viridis', density=2)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.colorbar(label='Speed [m/s]')
    plt.title('Streamlines')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
