#%%

# Imports

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
%matplotlib widget

from scipy.stats.qmc import Sobol

# from juliacall import Main
# Main.include("SPH.jl")
from SPH_core import Kernell, Boundary, Fields, Physics, TimeIntegration

# %%
solver_settings = {"h": 0.1, 
                   "n_particles": 100, 
                   "dt": 0.001, 
                   "t_max": 1.0, 
                   'gamma':1.4, 
                   "rho":1, 
                   "m":0.001}

# %%
#==================================
# Static test
#==================================

# Initialise kernel functions
kernel      = Kernell('cubicspline', solver_settings['h'])

# Create boundaryies
boundary    = Boundary()

# Initialise property fields, physics equations and integrator
fields      = Fields(kernel, solver_settings)
physics     = Physics(solver_settings['gamma'], boundary)
itegr       = TimeIntegration(solver_settings['dt'], fields, physics, boundary)

# Create coordinates
x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
x, y = np.meshgrid(x, y)
X = np.vstack((x.flatten(),y.flatten())).T

# Initialise density 
Y = fields.density(X)
E = fields.psi['e']

# Plot initialized density
plt.figure()
sc = plt.pcolormesh(x,y,Y[:,0].reshape(100,100))
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
plt.colorbar(sc)

# Update 10 iteration steps and plot again
plt.figure()
for i in range(10):
    itegr.leapFrog()
Y = fields.density(X)
sc = plt.scatter(X[:,0],X[:,1], c=Y)
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
plt.colorbar(sc)

#%%
# Y = fields.velocity(X)[:,1]
Y = fields.grad_sph('rho',X)[:,0]

sc = plt.scatter(X[:,0],X[:,1], c=Y)
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
plt.colorbar(sc)

# %%
#==================================
# Time integration test
#==================================
class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=80):
        self.numpoints = numpoints
        self.stream = self.data_stream()

        # Initialize the SPH solver
        self.solver_settings = {"h": 0.1, "n_particles": 100, "dt": 0.1, "t_max": 1.0, 'gamma':1.4, "rho":1.169}
        self.solver_settings['m'] = self.solver_settings['rho']/(self.solver_settings['n_particles']*(2*self.solver_settings['h'])**2)/900

        self.kernel     = Kernell('cubicspline', self.solver_settings['h'])
        self.boundary   = Boundary()
        self.fields     = Fields(self.kernel, self.solver_settings)
        self.physics    = Physics(self.solver_settings['gamma'], self.boundary)
        self.itegr      = TimeIntegration(self.solver_settings['dt'], self.fields, self.physics, self.boundary)
        # self.X = Sobol(2).random_base2(9)-0.5

        # Coordinates
        x = np.linspace(-1,1,self.numpoints)
        y = np.linspace(-1,1,self.numpoints)
        x, y = np.meshgrid(x, y)
        X = np.vstack((x.flatten(),y.flatten())).T
        self.X = [x, y, X]

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=10, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        Y = next(self.stream)
        self.scat = self.ax.pcolormesh(self.X[0], self.X[1], Y[:,0].reshape(self.numpoints,self.numpoints))
        # self.scat = self.ax.scatter(self.X[:,0],self.X[:,1], c=Y[:,0], cmap="jet", edgecolor="k")
        self.ax.axis([-0.5, 0.5, -0.5, 0.5])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        while True:
            self.itegr.leapFrog()
            Y = self.fields.density(self.X[2])
            yield Y

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set colors..
        self.scat.set_array(data[:,0])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

if __name__ == '__main__':
    a = AnimatedScatter()
    plt.show()

#%%
#==================================
# Time integration test
#==================================
class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=40):
        self.numpoints = numpoints
        self.stream = self.data_stream()
        self.framecount = 0

        # Initialize the SPH solver
        self.solver_settings = {"h": 0.1, 
                                "n_particles": 200, 
                                "dt": 0.1, 
                                "t_max": 1.0, 
                                'gamma':1.4, 
                                "rho":1.169}
        
        self.solver_settings['m'] = self.solver_settings['rho']/(self.solver_settings['n_particles']*(2*self.solver_settings['h'])**2)/1000

        self.kernel     = Kernell('cubicspline', self.solver_settings['h'])
        self.boundary   = Boundary()
        self.fields     = Fields(self.kernel, self.solver_settings)
        self.physics    = Physics(self.solver_settings['gamma'], self.boundary)
        self.itegr      = TimeIntegration(self.solver_settings['dt'], self.fields, self.physics, self.boundary)
        # self.X = Sobol(2).random_base2(9)-0.5

        self.X = self.fields.psi['x']

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=10,
                                          init_func=self.setup_plot, blit=True)
        # self.ani = animation.FuncAnimation(self.fig, self.update, interval=10,
        #                                   frames=80, repeat=False,
        #                                   init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        X, Y, E = next(self.stream)
        E_total = np.sum(E)
        self.scat = self.ax.scatter(X[:,0],X[:,1], c=E, cmap="jet", edgecolor="k")
        self.ax.axis([-0.5, 0.5, -0.5, 0.5])
        self.ax.set_title('{}'.format(E_total))
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        while True:
            self.itegr.leapFrog()
            X = self.fields.psi['x']
            Y = self.fields.psi['rho']
            # U = np.linalg.norm(self.fields.psi['u'], axis=1)
            E = self.fields.psi['e'][:,0]
            yield X, Y, E

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[0])

        # Set colors..
        self.scat.set_array(data[2])

        # Set title
        E = data[2]
        E_total = np.sum(E)
        self.ax.set_title('{}'.format(E_total))

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

if __name__ == '__main__':
    a = AnimatedScatter()
    plt.show()

# %%
