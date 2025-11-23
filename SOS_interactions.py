import numpy as np

def fully_connected_interactions(N, weight):
    return {i: {j: weight for j in range(N) if j != i} for i in range(N)}
def no_interactions(N):
    return {i: {} for i in range(N)}

#order is the number of neighbors in each direction (1 means 6 neighbors in 3D)
def topological_neighbor_interactions(lattice, order, weight):
    interactions = {}
    for i in range(lattice.x):
        for j in range(lattice.y):
            for k in range(lattice.z):
                ''' add neighbors in a cube like manner around each bird'''
                for di in range(-order, order + 1):
                    for dj in range(-order, order + 1):
                        for dk in range(-order, order + 1):
                            if abs(di) + abs(dj) + abs(dk) == 0 or abs(di) + abs(dj) + abs(dk) > order:
                                continue
                            ni = i + di
                            nj = j + dj
                            nk = k + dk
                            if 0 <= ni < lattice.x and 0 <= nj < lattice.y and 0 <= nk < lattice.z:
                                idx = i * (lattice.y * lattice.z) + j * lattice.z + k
                                nidx = ni * (lattice.y * lattice.z) + nj * lattice.z + nk
                                if idx not in interactions:
                                    interactions[idx] = {}
                                interactions[idx][nidx] = weight
    return interactions

