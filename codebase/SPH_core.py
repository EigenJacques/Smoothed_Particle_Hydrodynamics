#%%
#=======================================
# Imports
#=======================================
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.cm as cm

# from juliacall import Main
# Main.include("SPH.jl")

#%%
#=======================================
# Core code
#=======================================

class Boundary():
    """ 
    """ 

    def __init__(self) -> None:
        self.boundary_force         = 10
        self.boundary_sharpness     = 2

    def upper(self,X, a):
        return -X[:,1] + a

    def lower(self,X, a):
        return X[:,1] + a

    def left(self,X,  a):
        return X[:,0] + a

    def right(self,X, a):
        return -X[:,0] + a

    def square(self,X):

        n = self.boundary_sharpness

        x1 = self.upper(X,0.25)
        x2 = self.lower(X,0.25)
        x3 = self.left(X,0.25)
        x4 = self.right(X,0.25)
        x5 = (x1 + x2 - (x1**n + x2**n )**(1/n))
        x6 = (x3 + x4 - (x3**n + x4**n )**(1/n))

        return self.boundary_force*(x5 + x6 - (x5**n + x6**n)**(1/n))
    

class Kernell():
    """ Particle kernel class. 
    """ 

    def __init__(self, kern_type, h) -> None:

        self.h = h

        if kern_type == "cubicspline":
            self.k     = lambda x : self.cubicspline(*self.kernelValidation(x, self.h))
            self.dk    = lambda x : self.dcubicspline(*self.kernelValidation(x, self.h))

        elif kern_type == "gaussian":
            self.k     = lambda x : self.gaussian(*self.kernelValidation(x, self.h))
            self.dk    = lambda x : self.dgaussian(*self.kernelValidation(x, self.h))

        elif kern_type == 'triangular':
            self.k     = lambda x : self.triangular(*self.kernelValidation(x, self.h))
            self.dk    = lambda x : self.dtriangular(*self.kernelValidation(x, self.h))

        else:
            raise NotImplementedError(f" '{kern_type}' kernel not implemented")
        
    def kernelValidation(self, r, h):
        """ Performs tests on the inputs . 
        """ 

        assert np.all(r >= 0),  "r must be in [0,inf)"
        assert h > 0,           "h must be in (0,inf)"

        return r, h

    def triangular(self, r, h):
        """ Returns the triangular kernel value. 
        
        Arguments
        ----------
        r: float
            Distance between evaluation and kernel center
        h: float
            Smoothing length
        
        Parameters
        ----------
        
        Returns
        -------
        y: float
            Kernel value
        
        Notes
        ----- 
        This kernel is not smooth. 
        
        """ 

        x = r/h
        y = np.zeros_like(x)

        # Interval 1
        a = np.where((0 <= x )*( x < 1))
        y[a] = 1/h*(1 - x[a])

        # Interval 2
        c = np.where(1 <= x)
        y[c] = 0

        return y

    def dtriangular(self, r, h):
        """ Returns the derivative of the triangular kernel value. 
        
        Arguments
        ----------
        r: float
            Distance between evaluation and kernel center
        h: float
            Smoothing length
        
        Parameters
        ----------
        
        Returns
        -------
        dy: float
            Kernel value
        
        Notes
        ----- 
        This kernel is not smooth. 
        
        """ 
        
        x = r/h
        dy = np.zeros_like(x)

        # Interval 1
        a = np.where((0 <= x )*( x < 1))
        dy[a] = -1

        # Interval 2
        c = np.where(1 <= x)
        dy[c] = 0

        return dy

    def gaussian(self, r, h):
        x = r/h
        y = 1/(np.pi**(3/2)*h**3)*np.exp(-x**2)

        return y

    def dgaussian(self, r, h):
        x = r/h
        y = -2*x*np.exp(-x**2)

        return y

    def cubicspline(self, r, h):
        
        x = r/h
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
    

class Fields():
    """ 
    """

    def __init__(self, kernel:object, solver_settings:dict) -> None:
        ''' 
        Arguments
        ----------
        * kernel
        * sover_settings

        Parameters
        ----------
        psi:    x :     coorinate locations of particles
                rho:    density of particle
                m:      mass of particle
                u:      velocity of particle
                e:      internal energy of particle
                mt:     

        Returns
        -------

        
        Notes
        ----- 
        '''

        self.psi = { 'x':    0.5*np.random.random((solver_settings['n_particles'],2))-0.25,
                     'rho':  solver_settings['rho']*np.ones((solver_settings['n_particles'], 1)),
                     'm':    solver_settings['m']*np.ones((solver_settings['n_particles'], 1)),
                     'u':    np.zeros((solver_settings['n_particles'],2)),
                     'e':    0.1*np.ones((solver_settings['n_particles'], 1)),
                     'mt':   np.ones((solver_settings['n_particles'], 1))}
        self.kernel = kernel

    def sph(self, fld, xi):
        """ Return a vector field, apporximated using the SPH discretization. 
        
        Arguments
        ----------
        fld: str
            Field to be evaluated
        xi: ndarray
            Evaluation points
        
        Parameters
        ----------
        
        Returns
        -------
        phi: ndarray
            Vector field
        
        Notes
        ----- 
        
        """ 
        
        m, rho, x, fld = self.psi['m'],self.psi['rho'],self.psi['x'],self.psi[fld]

        phi = []

        for xii in xi:
            k = self.kernel.k(np.linalg.norm(xii[np.newaxis,:]-x, axis=1))[:,np.newaxis]
            phi.append(np.sum(m/rho*fld*k, axis=0))

        return np.array(phi)

    def grad_sph(self, fld, xi):
        """ Return the gradient of a scalar field, approximated using the SPH discretization . 
        """ 
        n = xi/np.linalg.norm(xi, axis=1)[:,None]
        m, rho, x, fld = self.psi['m'],self.psi['rho'],self.psi['x'],self.psi[fld]

        assert fld.shape[1] == 1, "fld must be a scalar field"

        phi = []
        for xii, nii in zip(xi, n):
            dk = np.tile(self.kernel.dk(np.linalg.norm(xii[np.newaxis,:]-x, axis=1))[:,np.newaxis],(1, 2))*nii
            phi.append(np.sum(m/rho*fld*dk, axis=0))

        return np.array(phi)

    def div_sph(self, fld, xi):
        """ Return the divergence of a vector field, approximated using the SPH discretization . 
        """

        n = xi/np.linalg.norm(xi, axis=1)[:,None]
        phi = 0.0
        for m, rho, x, fld in zip(self.psi['m'],self.psi['rho'],self.psi['x'],self.psi[fld]):
            fld     = fld[np.newaxis,:]
            dkern    = np.tile(self.kernel.dk(np.linalg.norm(xi-x, axis=1))[np.newaxis,:],(2,1)).T*n
            phi     = phi + (m/rho) * fld @ (dkern.T)

        return phi

    # def curl_sph(self, fld, xi):
        """ Return the curl of a vector field, approximated using the SPH discretization . 
        """

    #     phi = 0.0
    #     for m, rho, x, fld in zip(self.psi['m'],self.psi['rho'],self.psi['x'],self.psi[fld]):

    #         phi = phi + m/rho*fld*self.kernel.self.kernel(np.linalg.norm(xi-x, axis=1))

    #     return phi

    def density(self,xi):
        rho = self.sph('rho',xi)
        return rho

    def velocity(self,xi):
        u = self.sph('u',xi)
        return u
    
    def intenergy(self,xi):
        u = self.sph('e',xi)
        return u


class Physics():
    """ 
    """

    def __init__(self, gamma, boundary)-> None:
        self.gamma = gamma
        self.boundary = boundary

    def materialDerivative(self, x, rho, e, v, dW, D, mi, vi, ei, rhoi):
        """ Returns the acceleration vector for each particle . 
        
        Arguments
        ----------
        
        Parameters
        ----------
        delta : float
            finite difference step size
        epsilon : float 
            wall hardness

        Returns
        -------
        
        Notes
        ----- 
        a = F/m = d/dt(u) = du/dt + u*grad(u)
        
        """

        delta = 0.001
        epsilon = 0.1

        # a = np.array([[0,1]])*-9.81 # Gravity

        du = -np.sum(mi*(((self.gamma-1)*e)/rho + ((self.gamma-1)*ei)/rhoi)*dW, axis=0)

        # Boundary forces
        if D[0] <= 0:
            dDx = (self.boundary.square(x[:, np.newaxis].T + delta*np.array([[1,0]])) - D)/delta
            dDy = (self.boundary.square(x[:, np.newaxis].T + delta*np.array([[0,1]])) - D)/delta
            dD = np.array([dDx, dDy])[:,0]

            # du = du                                   # no boundary force
            du = du + dD*epsilon*-np.min([0, D[0]])     # add boundary force

        return du
    
    def energyConservation(self, rho, e, v, dW, mi, vi):
        """ Description . 
        
        Arguments
        ----------
        
        Parameters
        ----------
        
        Returns
        -------
        
        Notes
        ----- 
        
        """ 

        de = ((self.gamma-1)*e)/rho*np.sum(mi*(vi - v)@dW)

        return de


class TimeIntegration():
    """ Time integration of the SPH equations.
    """ 

    def __init__(self, time_step, fields, physics, boundary) -> None:
        self.time_step = time_step
        self.fields = fields

        self.physics = physics
        self.boundary = boundary

    def leapFrog(self):
        """ 

        """ 

        # Field vectors
        m   = self.fields.psi['m']
        x   = self.fields.psi['x']
        rho = self.fields.psi['rho']
        u   = self.fields.psi['u']
        e   = self.fields.psi['e']

        # Kernel
        # W   = self.fields.sph("mt",x)
        dW  = self.fields.grad_sph("mt",x)

        # Boundaryies
        D = self.boundary.square(x)[:, np.newaxis]

        du = []
        de = []
        for xj, rhoj, uj, ej, mj, dWj, Dj in zip(x, rho, u, e, m, dW, D):
            de.append(self.physics.energyConservation(rhoj, ej, uj, dWj, m, u))
            du.append(self.physics.materialDerivative(xj, rhoj, ej, uj, dWj, Dj, m, u, e, rho))
        du = np.array(du)
        de = np.array(de)

        x = x + self.time_step*u
        u = u + self.time_step*du
        e = e + self.time_step*de

        self.fields.psi['x'] = x
        self.fields.psi['u'] = u
        self.fields.psi['e'] = e


# %%
