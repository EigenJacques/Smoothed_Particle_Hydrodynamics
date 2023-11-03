#%%

# Imports

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.cm     as cm

from juliacall import Main
Main.include("SPH.jl")
from SPH_core import Kernell, Boundary, Fields

# %%
solver_settings = {"h": 0.1,"n_particles": 50, "dt": 0.01, "t_max": 1.0}
kernel = Kernell('triangular', 0.2)
fields = Fields(kernel)

# %%
plt.figure()
X = np.random.random((10_000,2))-0.5
Y = fields.velocity(X)[:,0]
# Y = fields.div_sph('u',X)

sc = plt.scatter(X[:,0],X[:,1], c=Y)
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
plt.colorbar(sc)

# %%
