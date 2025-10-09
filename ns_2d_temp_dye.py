"""
@author: Swayam Kuckreja
@purpose: 2D incompressible Navier-Stokes with temperature as passive scalar
@date: 02-10-2025
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Grid parameters
NX = 200  # number of nodes in x direction (includes boundary points)
NY = 200  # number of nodes in y direction
LX = 1.0  # domain length in x direction [m]
LY = 1.0  # domain length in y direction [m]

# Time parameters
N_TIMESTEPS = 2000
N_PRESSURE_ITERATIONS = 50  # inner Poisson iterations for pressure
TIME_STEP_SAFETY_FACTOR = 0.2  # More conservative safety factor

# Physical parameters
KINEMATIC_VISCOSITY = 1.0e-4   # Higher viscosity for stability [m²/s]
RHO = 1.225  # density [kg/m³]

# Thermal properties (for air at ~300K)
THERMAL_CONDUCTIVITY = 0.26    # k [W/(m·K)]
SPECIFIC_HEAT = 1005.0          # cp [J/(kg·K)]
THERMAL_DIFFUSIVITY = THERMAL_CONDUCTIVITY / (RHO * SPECIFIC_HEAT)  # α = k/(ρ·cp) [m²/s]

# Boundary condition values
WALL_VELOCITY = 1.0  # reduced moving wall velocity [m/s]
INLET_VELOCITY = 1.0  # reduced inlet velocity [m/s]
INLET_TEMPERATURE = 350.0  # hot inlet temperature [K]
WALL_TEMPERATURE = 300.0  # wall temperature [K]
INITIAL_TEMPERATURE = 300.0  # initial domain temperature [K]

# Simulation type (set exactly one to True)
SIM1 = False  # moving wall simulation
SIM2 = True   # inlet-outlet flow simulation


def main():
    """Main simulation function"""
    
    # ========================================================================
    # GRID SETUP
    # ========================================================================
    
    # Calculate grid spacing
    DX = LX / (NX - 1)
    DY = LY / (NY - 1)
    
    # Create coordinate arrays
    x = np.linspace(0, LX, NX)
    y = np.linspace(0, LY, NY)
    X, Y = np.meshgrid(x, y)
    
    # ========================================================================
    # FIELD INITIALIZATION
    # ========================================================================
    
    # Initialize velocity, pressure, and temperature fields
    u = np.zeros_like(X)  # x-velocity [m/s]
    v = np.zeros_like(X)  # y-velocity [m/s]
    p = np.zeros_like(X)  # pressure [Pa]
    T = np.full_like(X, INITIAL_TEMPERATURE)  # temperature [K]
    
    u_hist = []
    v_hist = []
    T_hist = []

    # ========================================================================
    # FINITE DIFFERENCE OPERATORS
    # ========================================================================
    
    def central_diff_x(w):
        """Central difference in x-direction: ∂w/∂x"""
        diff = np.zeros_like(X)
        diff[1:-1, 1:-1] = (w[1:-1, 2:] - w[1:-1, 0:-2]) / (2 * DX)
        return diff
    
    def central_diff_y(w):
        """Central difference in y-direction: ∂w/∂y"""
        diff = np.zeros_like(X)
        diff[1:-1, 1:-1] = (w[2:, 1:-1] - w[0:-2, 1:-1]) / (2 * DY)
        return diff
    
    def laplace_x_component(w):
        """Second derivative in x-direction: ∂²w/∂x²"""
        diff = np.zeros_like(X)
        diff[1:-1, 1:-1] = (w[1:-1, 2:] - 2*w[1:-1, 1:-1] + w[1:-1, 0:-2]) / (DX * DX)
        return diff
     
    def laplace_y_component(w):
        """Second derivative in y-direction: ∂²w/∂y²"""
        diff = np.zeros_like(X)
        diff[1:-1, 1:-1] = (w[2:, 1:-1] - 2*w[1:-1, 1:-1] + w[0:-2, 1:-1]) / (DY * DY)
        return diff
    
    # ========================================================================
    # TIME STEP CALCULATION
    # ========================================================================
    
    # Calculate stable time step based on CFL and diffusion conditions
    # CFL condition: dt < dx/(2*u_max)
    # Diffusion condition: dt < dx²/(4*ν) for explicit schemes
    
    u_max = max(INLET_VELOCITY, WALL_VELOCITY)
    dt_cfl = TIME_STEP_SAFETY_FACTOR * min(DX, DY) / (2 * u_max)
    dt_diffusion = TIME_STEP_SAFETY_FACTOR * min(DX, DY)**2 / (4 * KINEMATIC_VISCOSITY)
    dt_thermal = TIME_STEP_SAFETY_FACTOR * min(DX, DY)**2 / (4 * THERMAL_DIFFUSIVITY)
    
    # Use the most restrictive time step
    dt = min(dt_cfl, dt_diffusion, dt_thermal)
    
    # Additional safety check - cap maximum time step
    dt = min(dt, 1e-4)
    
    # Calculate dimensionless numbers for stability assessment
    Re = u_max * min(LX, LY) / KINEMATIC_VISCOSITY
    Pe = u_max * min(LX, LY) / THERMAL_DIFFUSIVITY
    CFL = u_max * dt / min(DX, DY)
    
    print(f"Grid size: {NX} x {NY}")
    print(f"Grid spacing: dx = {DX:.4f} m, dy = {DY:.4f} m")
    print(f"Time step: {dt:.2e} s (CFL={CFL:.3f})")
    print(f"Reynolds number: {Re:.1f}")
    print(f"Péclet number: {Pe:.1f}")
    print(f"Total simulation time: {dt * N_TIMESTEPS:.4f} s")
    
    # Stability warnings
    if CFL > 0.5:
        print("WARNING: CFL > 0.5, simulation may be unstable!")
    if Re > 100:
        print("WARNING: High Reynolds number, consider finer grid or higher viscosity!")
    
    print("Starting simulation...\n")
    
    # ========================================================================
    # BOUNDARY CONDITION FUNCTIONS
    # ========================================================================
    def apply_bc_moving_wall(u, v, p, T):
        # dirichlet walls
        u[-1, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        u[0, :]  = WALL_VELOCITY
        
        v[-1, :] = 0.0 # top wall
        v[:, 0] = 0.0 # left wall
        v[:, -1] = 0.0 # right wall
        v[0, :] = 0.0 # bottom wall
        
        p[:, 0] = p[:, 1] # neumann on static walls
        p[:, -1] = p[:, -2]
        p[-1, :] = p[-2, :]
        p[0, :] = 0.0 # bottom moving wall dirichlet
        
        # temperature boundary conditions
        T[-1, :] = WALL_TEMPERATURE  # top wall
        T[:, 0] = WALL_TEMPERATURE   # left wall
        T[:, -1] = WALL_TEMPERATURE  # right wall
        T[0, :] = WALL_TEMPERATURE   # bottom moving wall (hot)
        
        return u, v, p, T
    
    def apply_bc_inlet_outlet(u, v, p, T):
        # inlet
        u[:, 0] = INLET_VELOCITY
        v[:, 0] = 0.0
        p[:, 0] = p[:, 1] # neumann: ∂p/∂x = 0
        T[:, 0] = INLET_TEMPERATURE  # hot inlet

        # outlet1
        u[:, -1] = u[:, -2] # neumann: ∂u/∂x = 0
        v[:, -1] = v[:, -2] # neumann: ∂v/∂x = 0
        p[:, -1] = 0.0 # dirichlet: fixed pressure (reference)
        T[:, -1] = T[:, -2]  # neumann: ∂T/∂x = 0 (convective outflow)

        # walls
        u[0, :] = 0.0 # bottom wall
        u[-1, :] = 0.0 # top wall
        v[0, :] = 0.0
        v[-1, :] = 0.0
        p[0, :] = p[1, :] # neumann: ∂p/∂y = 0
        p[-1, :] = p[-2, :] # neumann
        T[0, :] = WALL_TEMPERATURE  # bottom wall temperature
        T[-1, :] = WALL_TEMPERATURE  # top wall temperature
        
        # center box of walls for fun
        BOX = False
        TANK = True
        if BOX == True:
            u[int(np.floor(NY/3)):int(np.floor(2*NY/3)),int(np.floor(2*NX/5)):int(np.floor(3*NX/5))] = 0.0
            v[int(np.floor(NY/3)):int(np.floor(2*NY/3)),int(np.floor(2*NX/5)):int(np.floor(3*NX/5))] = 0.0
            p[int(np.floor(NY/3)):int(np.floor(2*NY/3)),int(np.floor(2*NX/5)):int(np.floor(3*NX/5))] = 0.0
            T[int(np.floor(NY/3)):int(np.floor(2*NY/3)),int(np.floor(2*NX/5)):int(np.floor(3*NX/5))] = WALL_TEMPERATURE
        if TANK == True:
            nozzle_width = int(np.floor(NY/3))
            u[0:nozzle_width, 0:nozzle_width] = 0
            v[0:nozzle_width, 0:nozzle_width] = 0
            p[0:nozzle_width, 0:nozzle_width] = 0
            T[0:nozzle_width, 0:nozzle_width] = WALL_TEMPERATURE
            u[-nozzle_width:, 0:nozzle_width] = 0
            v[-nozzle_width:, 0:nozzle_width] = 0
            p[-nozzle_width:, 0:nozzle_width] = 0
            T[-nozzle_width:, 0:nozzle_width] = WALL_TEMPERATURE
        return u, v, p, T

    # forward euler time marching
    
    for timestep in range(N_TIMESTEPS):
        
        # --------------------------------------------------------------------
        # Step 1: Calculate derivatives for momentum and energy equations
        # --------------------------------------------------------------------
        
        # Velocity derivatives for momentum equations
        du_dx = central_diff_x(u)
        du_dy = central_diff_y(u)
        dv_dx = central_diff_x(v)
        dv_dy = central_diff_y(v)
        d2u_dx2 = laplace_x_component(u)
        d2u_dy2 = laplace_y_component(u)
        d2v_dx2 = laplace_x_component(v)
        d2v_dy2 = laplace_y_component(v)
        
        # Temperature derivatives for energy equation
        dT_dx = central_diff_x(T)
        dT_dy = central_diff_y(T)
        d2T_dx2 = laplace_x_component(T)
        d2T_dy2 = laplace_y_component(T)
        
        # --------------------------------------------------------------------
        # Step 2: Solve momentum equations (predictor step)
        # --------------------------------------------------------------------
        
        # Momentum equations: ∂u/∂t + u·∇u = -∇p/ρ + ν∇²u
        # Add overflow protection and clipping
        
        # Calculate advection terms with overflow protection
        advection_u = np.clip(-u * du_dx - v * du_dy, -1e6, 1e6)
        advection_v = np.clip(-u * dv_dx - v * dv_dy, -1e6, 1e6)
        
        # Calculate diffusion terms
        diffusion_u = KINEMATIC_VISCOSITY * (d2u_dx2 + d2u_dy2)
        diffusion_v = KINEMATIC_VISCOSITY * (d2v_dx2 + d2v_dy2)
        
        # Clip diffusion terms to prevent overflow
        diffusion_u = np.clip(diffusion_u, -1e6, 1e6)
        diffusion_v = np.clip(diffusion_v, -1e6, 1e6)
        
        # Update velocities with overflow protection
        u_star = u + dt * (advection_u + diffusion_u)
        v_star = v + dt * (advection_v + diffusion_v)
        
        # Clip velocities to reasonable bounds
        u_star = np.clip(u_star, -10 * INLET_VELOCITY, 10 * INLET_VELOCITY)
        v_star = np.clip(v_star, -10 * INLET_VELOCITY, 10 * INLET_VELOCITY)
        
        # Check for NaN or infinite values
        if np.any(np.isnan(u_star)) or np.any(np.isinf(u_star)):
            print("ERROR: NaN or Inf detected in u_star! Stopping simulation.")
            break
        if np.any(np.isnan(v_star)) or np.any(np.isinf(v_star)):
            print("ERROR: NaN or Inf detected in v_star! Stopping simulation.")
            break
        
        # --------------------------------------------------------------------
        # Step 3: Solve energy equation (passive scalar)
        # --------------------------------------------------------------------
        
        # Energy equation: ρcp[∂T/∂t + u·∇T] = k∇²T
        # Simplified form: ∂T/∂t + u·∇T = α∇²T where α = k/(ρcp)
        # This is the advection-diffusion equation for temperature
        # Add overflow protection
        
        # Calculate advection and diffusion terms
        advection_T = np.clip(-u * dT_dx - v * dT_dy, -1e6, 1e6)
        diffusion_T = np.clip(THERMAL_DIFFUSIVITY * (d2T_dx2 + d2T_dy2), -1e6, 1e6)
        
        # Update temperature with bounds checking
        T_new = T + dt * (advection_T + diffusion_T)
        
        # Keep temperature within reasonable physical bounds
        T_new = np.clip(T_new, 200.0, 500.0)  # 200K to 500K
        
        # Check for NaN or infinite values
        if np.any(np.isnan(T_new)) or np.any(np.isinf(T_new)):
            print("ERROR: NaN or Inf detected in T_new! Stopping simulation.")
            break
        
        # --------------------------------------------------------------------
        # Step 4: Apply boundary conditions to intermediate fields
        # --------------------------------------------------------------------
        
        if SIM1:
            u_star, v_star, _, T_new = apply_bc_moving_wall(u_star, v_star, p, T_new)
        elif SIM2:
            u_star, v_star, _, T_new = apply_bc_inlet_outlet(u_star, v_star, p, T_new)
        else: 
            raise ValueError("No simulation type selected! Set SIM1 or SIM2 to True.")
        
        # --------------------------------------------------------------------
        # Step 5: Solve pressure Poisson equation (corrector step)
        # --------------------------------------------------------------------
        
        # Right-hand side: ρ/Δt * ∇·u*
        rhs = RHO / dt * (central_diff_x(u_star) + central_diff_y(v_star))
        
        # Iterative solution of ∇²p = rhs
        for iteration in range(N_PRESSURE_ITERATIONS):
            p_new = np.zeros_like(p)
            
            # Jacobi iteration: solve discrete Poisson equation
            denominator = -2 / (DX * DX) - 2 / (DY * DY)
            p_new[1:-1, 1:-1] = (
                rhs[1:-1, 1:-1] - 
                (p[1:-1, 2:] + p[1:-1, 0:-2]) / (DX * DX) - 
                (p[2:, 1:-1] + p[0:-2, 1:-1]) / (DY * DY)
            ) / denominator
            
            # Apply pressure boundary conditions
            if SIM1:
                _, _, p_new, _ = apply_bc_moving_wall(u_star, v_star, p_new, T_new)
            elif SIM2:
                _, _, p_new, _ = apply_bc_inlet_outlet(u_star, v_star, p_new, T_new)
            
            p = p_new
        
        # --------------------------------------------------------------------
        # Step 6: Velocity correction using pressure gradient
        # --------------------------------------------------------------------
        
        # Correct velocities: u^(n+1) = u* - Δt/ρ * ∇p^(n+1)
        u_new = u_star - dt / RHO * central_diff_x(p_new)
        v_new = v_star - dt / RHO * central_diff_y(p_new)
        
        # --------------------------------------------------------------------
        # Step 7: Apply final boundary conditions and update fields
        # --------------------------------------------------------------------
        
        if SIM1:
            u_new, v_new, _, _ = apply_bc_moving_wall(u_new, v_new, p, T_new)
        elif SIM2:
            u_new, v_new, _, _ = apply_bc_inlet_outlet(u_new, v_new, p, T_new)
        
        # Update solution fields
        u, v, p, T = u_new, v_new, p_new, T_new
        u_hist.append(u)
        v_hist.append(v)
        T_hist.append(T)

        # Progress output with stability monitoring
        if (timestep + 1) % 100 == 0:
            T_max, T_min = np.max(T), np.min(T)
            v_max = np.max(np.sqrt(u**2 + v**2))
            u_max_current = np.max(np.abs(u))
            v_max_current = np.max(np.abs(v))
            
            print(f"Step {timestep + 1:3d}/{N_TIMESTEPS}: "
                  f"T = {T_min:.1f}-{T_max:.1f}K, |V|_max = {v_max:.3f} m/s")
            
            # Check for instability
            if v_max > 20 * INLET_VELOCITY:
                print(f"WARNING: Velocities growing large! Max velocity = {v_max:.2f} m/s")
            if u_max_current > 50 * INLET_VELOCITY or v_max_current > 50 * INLET_VELOCITY:
                print("ERROR: Simulation became unstable! Stopping.")
                break
    
    print("\nSimulation completed!")
    

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    def show_final_state():
        # Create side-by-side plots for velocity and temperature
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subsample data for cleaner visualization (every 2nd point)
        X_plot = X
        Y_plot = Y
        u_plot = u
        v_plot = v
        T_plot = T
        
        # Left plot: Velocity magnitude with streamlines
        velocity_magnitude = np.sqrt(u_plot**2 + v_plot**2)
        im1 = ax1.contourf(X_plot, Y_plot, velocity_magnitude, cmap="viridis", levels=20)
        ax1.set_title("Velocity Magnitude [m/s]", fontsize=14, fontweight='bold')
        ax1.set_xlabel("x [m]", fontsize=12)
        ax1.set_ylabel("y [m]", fontsize=12)
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label("Velocity [m/s]", fontsize=11)
        
        # Add streamlines (subsample more for clarity)
        X_stream = X
        Y_stream = Y
        u_stream = u
        v_stream = v
        ax1.streamplot(X_stream, Y_stream, u_stream, v_stream, 
                    color="white", density=1.5, arrowsize=1.2, linewidth=0.8)
        
        # Right plot: Temperature distribution
        im2 = ax2.contourf(X_plot, Y_plot, T_plot, cmap="coolwarm", levels=20)
        ax2.set_title("Temperature Distribution [K]", fontsize=14, fontweight='bold')
        ax2.set_xlabel("x [m]", fontsize=12)
        ax2.set_ylabel("y [m]", fontsize=12)
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label("Temperature [K]", fontsize=11)
        
        # Set axis properties for both plots
        for ax in [ax1, ax2]:
            ax.set_xlim(0, LX)
            ax.set_ylim(0, LY)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        # Add simulation info as text
        info_text = (f"Grid: {NX}×{NY}, dt = {dt:.1e} s\n"
                    f"Re ≈ {INLET_VELOCITY * LY / KINEMATIC_VISCOSITY:.0f}, "
                    f"Pe ≈ {INLET_VELOCITY * LY / THERMAL_DIFFUSIVITY:.0f}")
        fig.suptitle(f"2D Navier-Stokes with Heat Transfer\n{info_text}", 
                    fontsize=12, y=0.98)
        
        plt.tight_layout()
        plt.show()
        return _

    def animate(val_hist):    
        # --- Create animation ---
        fig, ax = plt.subplots()
        cax = ax.pcolormesh(X, Y, val_hist[0], shading='auto', cmap='viridis')
        fig.colorbar(cax)

        def update(frame):
            cax.set_array(val_hist[frame].ravel())
            ax.set_title(f"Time: {frame*dt:.4f} s")
            return cax,

        anim = FuncAnimation(fig, update, frames=len(val_hist), interval=50, blit=False)
        plt.show()
        return _
    

    show_final_state()
    animate(v_hist)

    return 0


if __name__ == "__main__":
    main()