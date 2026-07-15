import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu


class PoissonSolver:
    def __init__(self, grid):
        self.grid = grid
        self.A = None
        self.lu = None
        self._build_matrix()

    def _build_matrix(self):
        g = self.grid
        N = g.Nx * g.Ny
        A = lil_matrix((N, N))

        for j in range(g.Ny):
            for i in range(g.Nx):
                k = i + j * g.Nx

                if k == 1:
                    A[k, k] = 1

                elif i == 0 or i == g.Nx-1 or j == 0 or j == g.Ny-1:
                    A[k, k] = 1
                    if i == 0:            A[k, k + 1]    = -1
                    elif i == g.Nx - 1:   A[k, k - 1]    = -1
                    elif j == 0:          A[k, k + g.Nx] = -1
                    elif j == g.Ny - 1:   A[k, k - g.Nx] = -1

                else:
                    A[k, k]        = -(2/g.dx**2 + 2/g.dy**2)
                    A[k, k - 1]    =  1/g.dx**2
                    A[k, k + 1]    =  1/g.dx**2
                    A[k, k - g.Nx] =  1/g.dy**2
                    A[k, k + g.Nx] =  1/g.dy**2

        self.A = A.tocsc()          # csc richiesto da splu
        self.lu = splu(self.A)      # fattorizzazione UNA VOLTA

    def solve(self, rhs):
        g = self.grid
        b = rhs.ravel().copy()
        b[1] = 0
        p_vec = self.lu.solve(b)    # solo solve, niente rifattorizzazione
        p = p_vec.reshape((g.Ny, g.Nx))
        p -= p.mean()
        return p

    def test(self, operators, grid):
        rhs = np.zeros((grid.Ny, grid.Nx))
        rhs[1:-1, 1:-1] = 1.0

        b = rhs.ravel().copy()
        b[1] = 0   # <-- era b[0]=0, inconsistente con k==1 della matrice

        p_vec = self.lu.solve(b)

        print(f"p_vec ha NaN: {np.any(np.isnan(p_vec))}")
        print(f"p_vec ha Inf: {np.any(np.isinf(p_vec))}")
        print(f"p_vec range: [{p_vec.min():.4e}, {p_vec.max():.4e}]")

        alg_res = np.max(np.abs(self.A.dot(p_vec) - b))
        print(f"Residuo algebrico (A*p - b): {alg_res:.4e}")

        p = p_vec.reshape((grid.Ny, grid.Nx))
        p -= p.mean()

        r = operators.laplacian(p, grid) - rhs
        phys_res = np.max(np.abs(r[1:-1, 1:-1]))
        print(f"Residuo fisico (lap(p)-rhs): {phys_res:.4e}")

        return phys_res