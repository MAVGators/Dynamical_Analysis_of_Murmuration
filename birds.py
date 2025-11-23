import numpy as np

class Headings3D:
    def __init__(self, vectors):
        self.v = self._normalize(np.asarray(vectors, dtype=float))

    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return v / n

    def as_array(self):
        return self.v

    def dot(self, other):
        other_v = other.v if isinstance(other, Headings3D) else np.asarray(other)
        return np.einsum("ij,ij->i", self.v, other_v)

    def average(self):
        avg = self.v.mean(axis=0, keepdims=True)
        return Headings3D(self._normalize(avg))