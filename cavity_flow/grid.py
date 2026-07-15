import numpy as np
from dataclasses import dataclass

@dataclass
class Grid:
    Nx: int
    Ny: int
    Lx: float = 1.0
    Ly: float = 1.0

    def __post_init__(self):
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)
        
        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)

        self.X, self.Y = np.meshgrid(self.x, self.y)

    def __repr__(self):
        return (f"Grid(Nx = {self.Nx}, Ny = {self.Ny}, "
            f"Lx = {self.Lx}, Ly = {self.Ly}, "
            f"dx = {self.dx:.4f}, dy = {self.dy:.4f})")