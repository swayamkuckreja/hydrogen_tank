# 2D Explicit Finite Difference Solver for Subsonic Compressible Flow in a Rectangular Domain
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
GAMMA = 1.4
R = 287.05

# Grid parameters
Lx = 10.0  # length in x [m]
Ly = 5.0  # length in y [m]
Nx = 100  # grid points in x
Ny = 100   # grid points in y
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Time stepping
dt = 5e-6  # time step [s]
nt = 1000  # number of time steps

# Initial conditions (uniform subsonic flow)
P0 = 101325.0  # Pa
T0 = 300.0     # K
rho0 = P0 / (R * T0)  # Ideal gas law
u0 = 50.0      # m/s (x-direction)
v0 = 0.0       # m/s (y-direction)

# State variables: [rho, rho*u, rho*v, E]
U = np.zeros((Nx, Ny, 4))
U[:, :, 0] = rho0
U[:, :, 1] = rho0 * u0  # Set initial x-momentum everywhere
U[:, :, 2] = rho0 * v0  # Set initial y-momentum everywhere
U[:, :, 3] = P0 / (GAMMA - 1) + 0.5 * rho0 * (u0**2 + v0**2)

def primitive_vars(U):
    rho = U[:, :, 0]
    u = U[:, :, 1] / rho
    v = U[:, :, 2] / rho
    E = U[:, :, 3]
    # Pressure from total energy and ideal gas law
    p = (GAMMA - 1) * (E - 0.5 * rho * (u**2 + v**2))
    T = p / (rho * R)
    return rho, u, v, p, T

def flux_x(U):
    rho, u, v, p, _ = primitive_vars(U)
    Fx = np.zeros_like(U)
    Fx[:, :, 0] = rho * u
    Fx[:, :, 1] = rho * u**2 + p
    Fx[:, :, 2] = rho * u * v
    Fx[:, :, 3] = u * (U[:, :, 3] + p)
    return Fx

def flux_y(U):
    rho, u, v, p, _ = primitive_vars(U)
    Fy = np.zeros_like(U)
    Fy[:, :, 0] = rho * v
    Fy[:, :, 1] = rho * u * v
    Fy[:, :, 2] = rho * v**2 + p
    Fy[:, :, 3] = v * (U[:, :, 3] + p)
    return Fy

def apply_boundary(U):
    # Inlet (left): fixed primitive variables
    U[0, :, 0] = rho0
    U[0, :, 1] = rho0 * u0
    U[0, :, 2] = rho0 * v0
    U[0, :, 3] = P0 / (GAMMA - 1) + 0.5 * rho0 * (u0**2 + v0**2)
    # Outlet (right): zero-gradient
    U[-1, :, :] = U[-2, :, :]
    U[-1, :, 1] = 0.0 
    # Top/bottom: slip wall (v=0)
    # Bottom wall (j=0)
    U[:, 0, :] = U[:, 1, :] # zero-gradient for rho, rho*u, E (also temp indirectly)
    U[:, 0, 2] = 0.0  # v = 0 at bottom
    # Top wall (j=Ny-1)
    U[:, -1, :] = U[:, -2, :]
    U[:, -1, 2] = 0.0  # v = 0 at top
    return U

def main():
    global U
    wall_drag_coeff = 10.0  # Increase for stronger wall drag
    for n in range(nt):
        Fx = flux_x(U)
        Fy = flux_y(U)
        U_new = U.copy()
        # Only update interior points, not boundaries
        U_new[1:-1,1:-1] = 0.25 * (U[2:,1:-1] + U[:-2,1:-1] + U[1:-1,2:] + U[1:-1,:-2]) \
            - dt/(2*dx) * (Fx[2:,1:-1] - Fx[:-2,1:-1]) \
            - dt/(2*dy) * (Fy[1:-1,2:] - Fy[1:-1,:-2])
        # Artificial wall drag: damp u near top and bottom
        # Bottom wall (j=0,1)
        U_new[:,0:2,1] *= np.exp(-wall_drag_coeff*dt)
        # Top wall (j=-2,-1)
        U_new[:,-2:,1] *= np.exp(-wall_drag_coeff*dt)
        # Enforce boundary conditions after update
        U_new = apply_boundary(U_new)
        U = U_new
        # Debug: print mean and max velocity in the domain
        rho, u, v, p, T = primitive_vars(U)
        if n % 100 == 0 or n == nt - 1:
            print(f"Step {n+1}/{nt} | Mean u: {np.mean(u):.2f} | Max u: {np.max(u):.2f}")

    # Plot results
    rho, u, v, p, T = primitive_vars(U)
    plt.figure(figsize=(12, 8))
    for i, (data, title) in enumerate(zip([rho, u, p, T],
                                         ['Density [kg/m^3]', 'u-velocity [m/s]', 'Pressure [Pa]', 'Temperature [K]'])):
        plt.subplot(2,2,i+1)
        plt.contourf(X, Y, data, 50)
        plt.colorbar()
        plt.title(title)
    plt.tight_layout()
    plt.show()

    # Streamline plot
    plt.figure(figsize=(8, 4))
    plt.title('Streamlines')
    speed = np.sqrt(u**2 + v**2)
    lw = 2 * speed / speed.max()
    plt.streamplot(X.T, Y.T, u.T, v.T, color=speed.T, linewidth=lw.T, cmap='viridis', density=2)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.colorbar(label='Speed [m/s]')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
