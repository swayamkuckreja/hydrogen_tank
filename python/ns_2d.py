"""
@author: Swayam Kuckreja
@purpose: solve 2d incompressible Navier-Stokes for different BC's
@date: 01-06-2025
"""
import numpy as np
import matplotlib.pyplot as plt


# global problem parameters
NX = 300 # includes boundary points

NY = 100 # number of nodes in y direction
LX = 3.0 # domain length in x direction
LY = 1.0
N_TIMESTEPS = 100
N_PRESSURE_ITERATIONS = 100 # inner poisson iteration for pressure
TIME_STEP_SAFETY_FACTOR = 0.5
KINEMATIC_VISCOSITY = 1
RHO = 1.225
WALL_VELOCITY = 2.0
INLET_VELOCITY = 2.0

# set exactly one of these to true
SIM1 = False # moving wall simulation
SIM2 = True # inlet outlet flow

def main():
    DX = LX / (NX - 1)
    DY = LY / (NY - 1)
    x = np.linspace(0, LX, NX)
    y = np.linspace(0, LY, NY)
 
    X, Y = np.meshgrid(x,y)
    
    # initializing u, v and p
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    p = np.zeros_like(X)
    
    # defining finite difference derivatives
    def central_diff_x(w):
        diff = np.zeros_like(X)
        diff[1:-1, 1:-1] = (w[1:-1, 2:] - w[1:-1, 0:-2]) / (2 * DX)
        return diff
    
    def central_diff_y(w):
        diff = np.zeros_like(X)
        diff[1:-1, 1:-1] = (w[2:,1:-1] - w[0:-2, 1:-1]) / (2 * DY)
        return diff
    
    def laplace_x_component(w):
        diff = np.zeros_like(X)
        diff[1:-1, 1:-1] = (w[1:-1, 2:] - 2*w[1:-1, 1:-1] + w[1:-1, 0:-2]) / (DX * DX)
        return diff
     
    def laplace_y_component(w):
        diff = np.zeros_like(X)
        diff[1:-1, 1:-1] = (w[2:,1:-1] - 2*w[1:-1, 1:-1] + w[0:-2, 1:-1]) / (DY * DY)
        return diff
    
    # finding stable time step (based on yt video by Utah guy)
    dt = TIME_STEP_SAFETY_FACTOR * 0.5 * min(DX,DY) * min(DX, DY) / KINEMATIC_VISCOSITY
    # dt = 0.0001
    print(dt*N_TIMESTEPS)
    print(f"Simulation time: {dt * N_TIMESTEPS}")
    
    # functions that apply boundary conditions
    def apply_bc_moving_wall(u, v, p):
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
        return u, v, p
    
    def apply_bc_inlet_outlet(u, v, p):
        # inlet
        u[:, 0] = INLET_VELOCITY
        v[:, 0] = 0.0
        p[:, 0] = p[:, 1] # neumann: ∂p/∂x = 0

        # outlet
        u[:, -1] = u[:, -2] # neumann: ∂u/∂x = 0
        v[:, -1] = v[:, -2] # neumann: ∂v/∂x = 0
        p[:, -1] = 0.0 # dirichlet: fixed pressure (reference)

        # walls
        u[0, :] = 0.0 # bottom wall
        u[-1, :] = 0.0 # top wall
        v[0, :] = 0.0
        v[-1, :] = 0.0
        p[0, :] = p[1, :] # neumann: ∂p/∂y = 0
        p[-1, :] = p[-2, :] # neumann
        
        # center box of walls for fun
        BOX = False
        TANK = False
        if BOX == True:
            u[int(np.floor(NY/3)):int(np.floor(2*NY/3)),int(np.floor(2*NX/5)):int(np.floor(3*NX/5))] = 0.0
            v[int(np.floor(NY/3)):int(np.floor(2*NY/3)),int(np.floor(2*NX/5)):int(np.floor(3*NX/5))] = 0.0
            p[int(np.floor(NY/3)):int(np.floor(2*NY/3)),int(np.floor(2*NX/5)):int(np.floor(3*NX/5))] = 0.0
        if TANK == True:
            nozzle_width = int(np.floor(NY/3))
            u[0:nozzle_width, 0:nozzle_width] = 0
            v[0:nozzle_width, 0:nozzle_width] = 0
            p[0:nozzle_width, 0:nozzle_width] = 0
            u[-nozzle_width:, 0:nozzle_width] = 0
            v[-nozzle_width:, 0:nozzle_width] = 0
            p[-nozzle_width:, 0:nozzle_width] = 0
        return u, v, p

    # forward euler time marching
    for _ in range(N_TIMESTEPS):
        du_dx = central_diff_x(u)
        du_dy = central_diff_y(u)
        dv_dx = central_diff_x(v)
        dv_dy = central_diff_y(v)
        d2u_dx2 = laplace_x_component(u)
        d2u_dy2 = laplace_y_component(u)
        d2v_dx2 = laplace_x_component(v)
        d2v_dy2 = laplace_y_component(v)
        
        u_star = u + dt * (-u * du_dx -v * du_dy + KINEMATIC_VISCOSITY * (d2u_dx2 + d2u_dy2))
        v_star = v + dt * (-u * dv_dx -v * dv_dy + KINEMATIC_VISCOSITY * (d2v_dx2 + d2v_dy2))
        
        # enforce BC's on u_star and v_star before using them
        if SIM1:
            u_star, v_star, _ = apply_bc_moving_wall(u_star, v_star, p)
        elif SIM2:
            u_star, v_star, _ = apply_bc_inlet_outlet(u_star, v_star, p)
        else: 
            raise(Exception("No simulation set to run!"))
        
        # solving for p(n+1) using pressure poisson equation
        rhs = RHO / dt * (central_diff_x(u_star) + central_diff_y(v_star))
        for _ in range(N_PRESSURE_ITERATIONS):
            p_new = np.zeros_like(p)
            p_new[1:-1, 1:-1] = (rhs[1:-1, 1:-1] - ((p[1:-1, 2:] + p[1:-1, 0:-2]) / (DX * DX)) - ((p[2:, 1:-1] + p[0:-2, 1:-1]) / (DY * DY))) / ((-2 / (DX * DX)) - (2 / (DY * DY)))
            
            # enforce pressure BC's during each iteration step
            if SIM1:
                _, _, p_new = apply_bc_moving_wall(u_star, v_star, p_new)
            elif SIM2:
                _, _, p_new = apply_bc_inlet_outlet(u_star, v_star, p_new)
            else:
                raise(Exception("No simulation set to run!"))
            
            p = p_new
        
        # solving for final velocities u,v(n+1)
        u_new = u_star - dt / RHO * central_diff_x(p_new)
        v_new = v_star - dt / RHO * central_diff_y(p_new)
        
        # enforce velocity boundary conditions again
        if SIM1:
            u_new, v_new, _ = apply_bc_moving_wall(u_new, v_new, p)
        elif SIM2:
            u_new, v_new, _ = apply_bc_inlet_outlet(u_new, v_new, p)
        else:
            raise(Exception("No simulation set to run!"))
        
        u = u_new
        v = v_new
        p = p_new # last one can be left out but its fine
    
    # plotting 
    # the [::2, ::2] selects only every second entry (less cluttering plot)
    plt.figure()
    plt.contourf(X[::2, ::2], Y[::2, ::2], u[::2, ::2], cmap="coolwarm")
    plt.colorbar()

    #plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color="black")
    #plt.streamplot(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color="black")
    plt.xlim((0, LX))
    plt.ylim((0, LY))
    plt.show()
    return 0

if __name__ == "__main__":
    main()