# Basic 3D particle class with a tiny Vector3 helper.
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> Vector3:
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: Vector3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> Vector3:
        l = self.length()
        return self / l if l > 0 else Vector3(0.0, 0.0, 0.0)

    def copy(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)


class Particle:
    """
    Simple 3D particle with Euler integration.
    - position, velocity are Vector3
    - mass must be > 0
    - forces are accumulated each frame; call update(dt) to integrate and clear forces
    """

    def __init__(
        self,
        position: Vector3 | None = None,
        velocity: Vector3 | None = None,
        mass: float = 1.0,
        damping: float = 0.0,
    ):
        self.position = position.copy() if position else Vector3()
        self.velocity = velocity.copy() if velocity else Vector3()
        self.mass = max(1e-8, mass)
        self.inv_mass = 1.0 / self.mass
        self._force_acc = Vector3()
        self.damping = max(0.0, damping)  # simple velocity damping per second

    def apply_force(self, force: Vector3) -> None:
        """Accumulate a force (N)."""
        self._force_acc += force

    def clear_forces(self) -> None:
        self._force_acc = Vector3()

    def update(self, dt: float) -> None:
        """
        Integrate motion using explicit Euler:
        a = F / m
        v += a * dt
        apply damping: v *= (1 - damping*dt)  (clamped)
        pos += v * dt
        """
        if dt <= 0:
            return
        # acceleration
        accel = self._force_acc * self.inv_mass
        # integrate velocity
        self.velocity = self.velocity + accel * dt
        # damping
        factor = max(0.0, 1.0 - self.damping * dt)
        self.velocity = self.velocity * factor
        # integrate position
        self.position = self.position + self.velocity * dt
        # clear forces for next step
        self.clear_forces()

    def set_mass(self, mass: float) -> None:
        self.mass = max(1e-8, mass)
        self.inv_mass = 1.0 / self.mass

    def is_static(self) -> bool:
        return math.isinf(self.inv_mass) or self.mass <= 0

    def __repr__(self) -> str:
        return f"Particle(pos={self.position}, vel={self.velocity}, mass={self.mass})"


# Example usage when run as a script
if __name__ == "__main__":
    p = Particle(position=Vector3(0, 0, 0), velocity=Vector3(1, 2, 0), mass=2.0, damping=0.1)
    gravity = Vector3(0, -9.81, 0) * p.mass  # force = m * g
    dt = 0.016  # ~60 FPS

    for step in range(10):
        p.apply_force(gravity)
        p.update(dt)
        print(f"t={(step+1)*dt:.3f}s pos={p.position} vel={p.velocity}")