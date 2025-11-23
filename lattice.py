import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
For short time approximation use the fixed
lattice and place birds on verticies
'''

class Lattice:
    def __init__(self, x,y,z, spacing):
        self.x = x
        self.y = y
        self.z = z
        self.lattice = np.zeros(shape = (x,y,z), dtype=object)
        self.spacing = spacing

    def populate_lattice(self, data_points):
        if data_points.shape[0] != self.x * self.y * self.z:
            raise ValueError("data_points does not match lattice size")
        for i in range(self.x):
            for j in range(self.y):
                for k in range(self.z):
                    index = i * (self.y * self.z) + j * self.z + k
                    self.lattice[i, j, k] = data_points[index]

        
        


if __name__ == "__main__":
    # Choose number of lattice points per side (must be integer)
    lattice = Lattice(4,3,6, spacing=1.0)
    data = np.random.rand(4*3*6)
    lattice.populate_lattice(data)
    print(lattice.lattice)