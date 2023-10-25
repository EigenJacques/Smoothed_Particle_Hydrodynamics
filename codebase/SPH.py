#%%
import numpy as np
from matplotlib import pyplot as plt


#%%

class Boundary():
    """ 
    """ 

    def __init__(self, ) -> None:
        pass

    def upper(self,X, a):
        return -X[:,1] + a

    def lower(self,X, a):
        return X[:,1] + a

    def left(self,X,  a):
        return X[:,0] + a

    def right(self,X, a):
        return -X[:,0] + a

    def square(self,X):
        n = 2

        x1 = self.upper(X,0.5)
        x2 = self.lower(X,0.5)
        x3 = self.left(X,0.5)
        x4 = self.right(X,0.5)
        x5 = (x1 + x2 - (x1**n + x2**n )**(1/n))
        x6 = (x3 + x4 - (x3**n + x4**n )**(1/n))

        return x5 + x6 - (x5**n + x6**n)**(1/n)

#%%
x = np.linspace(-1,1,1000)
y = np.linspace(-1,1,1000)
x, y = np.meshgrid(x, y)

X = np.vstack((x.flatten(),y.flatten())).T

domain = Boundary().square(X)
domain = domain.reshape(x.shape)

plt.figure()
plt.contour(x,y,domain,levels=[0.0])

# %%
domain = Boundary()
%timeit domain.square(X)
# %%


