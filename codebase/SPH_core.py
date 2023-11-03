#%%
#=======================================
# Imports
#=======================================
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.cm     as cm

from juliacall import Main
Main.include("SPH.jl")

#%%
#=======================================
# Core code
#=======================================
class Kernell():
    """ 
    """
    def __init__(self, type, h) -> None:
        if type == "cubicspline":
            self.kernel = lambda x : self.cubicspline(x, h)
            self.dkernel = lambda x : self.dcubicspline(x, h)

        elif type == "gaussian":
            raise NotImplementedError("Gaussian kernel not implemented")
            self.kernel = lambda x : self.gaussian(x, h)
            self.dkernel = lambda x : self.dgaussian(x, h)

        elif type == 'triangular':
            self.kernel = lambda x : self.triangular(x, h)
            self.dkernel = lambda x : self.dtriangular(x, h)

        else:
            raise NotImplementedError("Kernel not implemented")

    def triangular(self, r, h):

        x = r/h
        assert np.all(x >= 0), "x must be in [0,inf)"
        assert h > 0, "h must be in (0,inf)"

        y = np.zeros_like(x)

        # Interval 1
        a = np.where((0 <= x )*( x < 1))
        y[a] = 1/h*(1 - x[a])

        # Interval 2
        c = np.where(1 <= x)
        y[c] = 0

        return y


    def dtriangular(self, r, h):

        x = r/h
        assert np.all(x >= 0), "x must be in [0,inf)"
        assert h > 0, "h must be in (0,inf)"

        y = np.zeros_like(x)

        # Interval 1
        a = np.where((0 <= x )*( x < 1))
        y[a] = -1

        # Interval 2
        c = np.where(1 <= x)
        y[c] = 0

        return y

    def gaussian(self, r, h):
        x = r/h
        assert np.all(x >= 0), "x must be in [0,inf)"
        assert h > 0, "h must be in (0,inf)"

        y = 1/(np.pi**(3/2)*h**3)*np.exp(-x**2)

        return y

    def dgaussian(self, r, h):
        x = r/h
        assert np.all(x >= 0), "x must be in [0,inf)"
        assert h > 0, "h must be in (0,inf)"

        y = -2*x*np.exp(-x**2)

        return y

    def cubicspline(self, r, h):
        
        x = r/h
        assert np.all(x >= 0), "x must be in [0,inf)"
        assert h > 0, "h must be in (0,inf)"

        y = np.zeros_like(x)

        # Interval 1
        a = np.where((0 <= x )*( x < 1))
        y[a] = 1/(np.pi*(h**3))*(1 - 3/2*x[a]**2 + 3/4*x[a]**3)

        # Interval 2
        b = np.where((1 <= x)*( x < 2))
        y[b] = 1/(np.pi*(h**3))*(1/4*(2-x[b])**3)

        # Interval 3
        c = np.where(2 <= x)
        y[c] = 0

        return y

    def dcubicspline(self, r, h):
        
        x = r/h
        assert np.all(x >= 0), "x must be in [0,inf)"
        assert h > 0, "h must be in (0,inf)"

        y = np.zeros_like(x)

        # Interval 1
        a = np.where((0 <= x )*( x < 1))
        y[a] = 1/np.pi/(h**4)*(9/4*x[a]**2 - 3*x[a])

        # Interval 2
        b = np.where((1 <= x)*( x < 2))
        y[b] = 1/np.pi/(h**4)*(-3/4*(2-x[b])**2)

        # Interval 3
        c = np.where(2 <= x)
        y[c] = 0

        return y