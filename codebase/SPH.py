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
solver_settings = {"h": 0.1,"n_particles": 50, "dt": 0.01, "t_max": 1.0}
kernel      = Kernell('triangular', 0.2)
fields      = Fields(kernel, solver_settings)
physics     = Physics()
itegr       = TimeIntegration(solver_settings['dt'], fields, physics)

# %%
#==================================
# Static test
#==================================
plt.figure()

x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
x, y = np.meshgrid(x, y)
X = np.vstack((x.flatten(),y.flatten())).T

Y = fields.density(X)
sc = plt.pcolormesh(x,y,Y[:,0].reshape(100,100))
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
plt.colorbar(sc)
#%%
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
    def __init__(self, numpoints=30):
        self.numpoints = numpoints
        self.stream = self.data_stream()

        # Initialize the SPH solver
        self.solver_settings = {"h": 0.1,"n_particles": 50, "dt": 0.01, "t_max": 1.0}
        self.kernel     = Kernell('triangular', 0.2)
        self.fields     = Fields(self.kernel, self.solver_settings)
        self.physics    = Physics()
        self.itegr      = TimeIntegration(self.solver_settings['dt'], self.fields, self.physics)
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
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
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

# %%
