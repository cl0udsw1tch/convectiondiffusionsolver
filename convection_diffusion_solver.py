import numpy as np
from scipy.sparse import csc_matrix
from typing import Tuple, List, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import RectBivariateSpline
from sympy import simplify, symbols
from linear_solver import solve



class ConvectionDiffusionSolver:
    
    L_x:        float=1,
    L_y:        float=1,
    n_rows:     int=10,
    n_cols:     int=10,
    params = {}
     
    (phi_x_0, phi_x_L, phi_y_0,  phi_y_L,  phi_E, phi_W, phi_N, phi_S, phi_P, gamma, u_x, u_y,  delta_x, delta_y) = symbols('\phi_{x_0}  phi_{x_L} \phi_{y_0} \phi_{y_L} \phi_E \phi_W \phi_N \phi_S \phi_P \Gamma u_x u_y \delta_x \delta_y')
    
    A_data: np.ndarray
    A_rows: np.ndarray
    A_cols: np.ndarray

    A: csc_matrix
    b: np.ndarray
    x: np.ndarray
    
    error: float
    
    
    def __init__(self, 
        L_x:        float=1,
        L_y:        float=1,
        n_rows:     int=10,
        n_cols:     int=10,
        phi_x_0:     float=100,
        phi_x_L:     float=0,
        phi_y_0:     float=0,
        phi_y_L:     float=100,
        gamma:       float=1,
        u_x:         float=2,
        u_y:         float=2
        ):
        
        self.L_x        = L_x
        self.L_y        = L_y
        self.n_rows     = n_rows
        self.n_cols     = n_cols
        
        self.params = {
            self.phi_x_0: phi_x_0,
            self.phi_x_L: phi_x_L,
            self.phi_y_0: phi_y_0,
            self.phi_y_L: phi_y_L,
            self.gamma: gamma,
            self.u_x: u_x,
            self.u_y: u_y,
            self.delta_x: L_x / n_cols,
            self.delta_y: L_y / n_rows
        }
        
        N = n_rows * n_cols
        num_nonzero = 5 * N - 2 * n_cols - 2 * n_rows
        print(f"Numnonzero: {num_nonzero}")

        self.A_data = np.zeros((num_nonzero)).astype(np.float64)
        self.A_rows = np.zeros((num_nonzero), dtype=np.int64)
        self.A_cols = np.zeros((num_nonzero), dtype=np.int64)

        self.b = np.zeros((n_rows * n_cols, 1)).astype(np.float64)
        
        
    
    def construct_problem(self):
        
        # +--------+-------------------------------+--------+
        # |        |                               |        |
        # |    G   |               H               |    I   |
        # |        |                               |        |
        # |--------|-------------------------------|--------|
        # |        |                               |        |
        # |        |                               |        |
        # |        |                               |        |
        # |        |                               |        |
        # |        |                               |        |
        # |    D   |               E               |    F   |
        # |        |                               |        |
        # |        |                               |        |
        # |        |                               |        |
        # |        |                               |        |
        # |--------|-------------------------------|--------|
        # |        |                               |        |
        # |   A    |               B               |   C    |
        # |        |                               |        |
        # +--------+-------------------------------+--------+
        


        self.setAExpr()
        self.setBExpr()
        self.setCExpr()
        self.setDExpr()
        self.setEExpr()
        self.setFExpr()
        self.setGExpr()
        self.setHExpr()
        self.setIExpr()
        
        
        idx = [0]
        self.set_arrays(self.A_expr, 0, idx)
        
        for i in range(1, self.n_cols - 1):
            self.set_arrays(self.B_expr, i, idx)

        self.set_arrays(self.C_expr, self.n_cols - 1, idx)
   
        for j in range(1, self.n_rows - 1):
            k = j * self.n_cols
            self.set_arrays(self.D_expr, k, idx)
            for i in range(1, self.n_cols - 1):
                k = i + j * self.n_cols
                self.set_arrays(self.E_expr, k, idx)
            k = (self.n_cols - 1) + j * self.n_cols
            self.set_arrays(self.F_expr, k, idx)
            
        k = (self.n_rows - 1) * self.n_cols
        self.set_arrays(self.G_expr, k, idx) 
        for i in range(1, self.n_cols - 1):
            k = i + (self.n_rows - 1) * self.n_cols
            self.set_arrays(self.H_expr, k, idx)
        k = self.n_rows * self.n_cols  - 1
        self.set_arrays(self.I_expr, k, idx)


        A_csc = csc_matrix(
            (self.A_data, (self.A_rows, self.A_cols)), 
            shape=(self.n_rows * self.n_cols, self.n_rows * self.n_cols)
            )

        self.A = A_csc

    
    def solve_problem(self) -> Tuple[np.ndarray, float]:
        
        x = solve(self.A, self.b)
        res = np.linalg.norm(self.b - self.A.dot(np.expand_dims(x, axis=1)))
        
        self.error = res
        self.x = x.reshape(self.n_rows, self.n_cols)[::-1]
        
        return x, res
    

    def plot(self, num_contours: int = 20, name: str = "heatmap_plot.png"):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        sns.heatmap(self.x, cmap="viridis", ax=ax, cbar=True)

        X, Y = np.meshgrid(np.arange(self.x.shape[1]) + 0.5, np.arange(self.x.shape[0]) + 0.5)
        ax.contour(X, Y, self.x, levels=num_contours, colors='cyan', linewidths=1)

        # Set axis ticks
        ax.set_xticks([0, self.n_cols])
        ax.set_xticklabels([0, self.L_x], rotation=0)
        ax.set_yticks([0, self.n_rows])
        ax.set_yticklabels([self.L_y, 0], rotation=0)

        # TOP (in figure coordinates)
        ax.text(0.5, 1.05, f'$\phi_B = {self.params[self.phi_y_L]}$', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, weight='bold') 

        # BOTTOM (in figure coordinates)
        ax.text(0.5, -0.05, f'$\phi_B = {self.params[self.phi_y_0]}$', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, weight='bold')  

        # LEFT (in figure coordinates)
        ax.text(-0.05, 0.5, f'$\phi_B = {self.params[self.phi_x_0]}$', 
                ha='center', va='center', rotation=90, transform=ax.transAxes, fontsize=12, weight='bold') 

        # RIGHT (in figure coordinates)
        ax.text(1.03, 0.5, f'$\phi_B = {self.params[self.phi_x_L]}$', 
                ha='center', va='center', rotation=90, transform=ax.transAxes, fontsize=12, weight='bold') 

        # u text (in figure coordinates)
        ax.text(0, -0.1, f'${{\\bf u}} = \\langle {self.params[self.u_x]}, {self.params[self.u_y]} \\rangle$'
, 
                ha='center', va='center', transform=ax.transAxes, fontsize=8, weight='bold')  
        # Residual text (in figure coordinates)
        ax.text(1, -0.1, f'$L^2 Residual = {np.round(self.error, 4)}$', 
                ha='center', va='center', transform=ax.transAxes, fontsize=8, weight='bold')  

        plt.savefig(name, dpi=300, bbox_inches="tight")

    
    
    def orderConv(self, exact_res: float = 320, coarse_res: float = 80, fine_res: float = 160) -> float:

        
        exact_solver = ConvectionDiffusionSolver(n_rows=exact_res, n_cols=exact_res, params=self.params)
        coarse_solver = ConvectionDiffusionSolver(n_rows=coarse_res, n_cols=coarse_res, params=self.params)
        fine_solver = ConvectionDiffusionSolver(n_rows=fine_res, n_cols=fine_res, params=self.params)
        
        exact_solver.construct_problem()
        coarse_solver.construct_problem()
        fine_solver.construct_problem()
        
        x_coarse, _ = coarse_solver.solve_problem()
        x_fine, _ = fine_solver.solve_problem()
        x_exact, _ = exact_solver.solve_problem()

        linsp_coarse = np.linspace(0, 1, coarse_res)
        linsp_fine = np.linspace(0, 1, fine_res)
        linsp_exact = np.linspace(0, 1, exact_res)

        x_coarse_interpolator = RectBivariateSpline(linsp_coarse, linsp_coarse, x_coarse.reshape(coarse_res, coarse_res))

        x_coarse_interp = x_coarse_interpolator(linsp_exact, linsp_exact)

        x_fine_interpolator = RectBivariateSpline(linsp_fine, linsp_fine, x_fine.reshape(fine_res, fine_res))

        x_fine_interp = x_fine_interpolator(linsp_exact, linsp_exact)

        err_coarse = np.linalg.norm(x_exact - x_coarse_interp.flatten()) / coarse_res
        err_fine = np.linalg.norm(x_exact - x_fine_interp.flatten()) / fine_res

        O = (np.log(np.abs(err_coarse / err_fine))) / (np.log((1 / coarse_res) / (1 / fine_res)))
        
        return O
    
    def set_arrays(self, expr, k, idx):
        
        a_P = float(expr.coeff(self.phi_P))
        a_W = float(-expr.coeff(self.phi_W))
        a_E = float(-expr.coeff(self.phi_E))
        a_S = float(-expr.coeff(self.phi_S))
        a_N = float(-expr.coeff(self.phi_N))

        S_U = - float(expr.subs({self.phi_S: 0 , self.phi_N: 0, self.phi_W: 0, self.phi_E:0, self.phi_P: 0}))
        
        print(f'\rSetting cell {k} with idx {idx}, a_E={a_E},a_W={a_W},a_S={a_S},a_N={a_N},a_P={a_P},S_U={S_U}', sep='', end='')
  
        self.A_data[idx[0]] = a_P
        self.A_rows[idx[0]] = k
        self.A_cols[idx[0]] = k
    
        
        self.b[k] = float(S_U)
        idx[0]+=1
        
        if a_E != 0:
            self.A_data[idx[0]] = -a_E
            self.A_rows[idx[0]] = k
            self.A_cols[idx[0]] = k+1
            idx[0]+=1
        if a_W != 0:
            self.A_data[idx[0]] = -a_W
            self.A_rows[idx[0]] = k
            self.A_cols[idx[0]] = k-1
            idx[0]+=1
        if a_S != 0:
            self.A_data[idx[0]] = -a_S
            self.A_rows[idx[0]] = k
            self.A_cols[idx[0]] = k-self.n_cols
            idx[0]+=1
        if a_N != 0:
            self.A_data[idx[0]] = -a_N
            self.A_rows[idx[0]] = k
            self.A_cols[idx[0]] = k+self.n_cols
            idx[0]+=1
            

        
        
        
    def setAExpr(self):
        
        if self.params[self.gamma]:
            self.A_expr = (
            self.u_x * ((self.phi_E + self.phi_P) / 2 - self.phi_x_0) + self.u_y * ((self.phi_N + self.phi_P) / 2 - self.phi_y_0) - 
            self.gamma * (
                (self.phi_E - self.phi_P) / self.delta_x - (self.phi_P - self.phi_x_0) / (self.delta_x / 2) + 
                (self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_y_0) / (self.delta_y / 2))
            
            )
        else:
            self.A_expr = self.u_x * ((self.phi_E - self.phi_P) / self.delta_x - (self.phi_P - self.phi_x_0) / (self.delta_x / 2)) + self.u_y * ((self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_y_0) / (self.delta_y / 2))
        
        self.A_expr = simplify(self.A_expr.subs(self.params))

       


    def setBExpr(self):
        
        if self.params[self.gamma]:
            self.B_expr = (
            self.u_x * ((self.phi_E + self.phi_P) / 2 - (self.phi_P + self.phi_W) / 2) + self.u_y * ((self.phi_N + self.phi_P) / 2 - self.phi_y_0) -  
            self.gamma * (
                (self.phi_E - self.phi_P) / self.delta_x - (self.phi_P - self.phi_W) / self.delta_x + 
                (self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_y_0) / (self.delta_y / 2))
            
            )
        else:
            self.B_expr = self.u_x * ((self.phi_E - self.phi_P) / self.delta_x - (self.phi_P - self.phi_W) / self.delta_x) + self.u_y * ((self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_y_0) / (self.delta_y / 2))
        
        self.B_expr = simplify(self.B_expr.subs(self.params))
        
        
        
        
    def setCExpr(self):
        
        if self.params[self.gamma]:
            self.C_expr = (
            self.u_x * (self.phi_x_L - (self.phi_P + self.phi_W) / 2) + self.u_y * ((self.phi_N + self.phi_P) / 2 - self.phi_y_0) - 
            self.gamma * (
                (self.phi_x_L - self.phi_P) / (self.delta_x / 2) - (self.phi_P - self.phi_W) / self.delta_x + 
                (self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_y_0) / (self.delta_y / 2))
            
            )
        else:
            self.C_expr = self.u_x * ((self.phi_x_L - self.phi_P) / (self.delta_x / 2) - (self.phi_P - self.phi_W) / self.delta_x) + self.u_y * ((self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_y_0) / (self.delta_y / 2))
        
        self.C_expr = simplify(self.C_expr.subs(self.params))
        
        
    
        
    def setDExpr(self):
        
        if self.params[self.gamma]:
            self.D_expr = (
            self.u_x * ((self.phi_E + self.phi_P) / 2 - self.phi_x_0) + self.u_y * ((self.phi_N + self.phi_P) / 2 - (self.phi_P + self.phi_S) / 2) - 
            self.gamma * (
                (self.phi_E - self.phi_P) / self.delta_x - (self.phi_P - self.phi_x_0) / (self.delta_x / 2) + 
                (self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_S) / self.delta_y)
            
            )
        else:
            self.D_expr = self.u_x * ((self.phi_E - self.phi_P) / (self.delta_x) - (self.phi_P - self.phi_x_0) / (self.delta_x / 2)) + self.u_y * ((self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_S) / (self.delta_y))
        
        self.D_expr = simplify(self.D_expr.subs(self.params))
        
        
    
        
    def setEExpr(self):
        
        if self.params[self.gamma]:
            self.E_expr = (
            self.u_x * ((self.phi_E + self.phi_P) / 2 - (self.phi_P + self.phi_W) / 2) + self.u_y * ((self.phi_N + self.phi_P) / 2 - (self.phi_P + self.phi_S) / 2) -
            self.gamma * (
                (self.phi_E - self.phi_P) / self.delta_x - (self.phi_P - self.phi_W) / self.delta_x + 
                (self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_S) / self.delta_y)
            
            )
        else:
            self.E_expr = self.u_x * ((self.phi_E - self.phi_P) / (self.delta_x) - (self.phi_P - self.phi_W) / (self.delta_x)) + self.u_y * ((self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_S) / (self.delta_y))
        
        self.E_expr = simplify(self.E_expr.subs(self.params))
        
        
    
        
    def setFExpr(self):
        
        if self.params[self.gamma]:
            self.F_expr = (
            self.u_x * (self.phi_x_L - (self.phi_P + self.phi_W) / 2) + self.u_y * ((self.phi_N + self.phi_P) / 2 - (self.phi_P + self.phi_S) / 2) - 
            self.gamma * (
                (self.phi_x_L - self.phi_P) / (self.delta_x / 2) - (self.phi_P - self.phi_W) / self.delta_x + 
                (self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_S) / self.delta_y)
            )
        else:
            self.F_expr = self.u_x * ((self.phi_x_L - self.phi_P) / (self.delta_x / 2) - (self.phi_P - self.phi_W) / (self.delta_x)) + self.u_y * ((self.phi_N - self.phi_P) / self.delta_y - (self.phi_P - self.phi_S) / (self.delta_y))
        
        self.F_expr = simplify(self.F_expr.subs(self.params))
        
        
        
        
    def setGExpr(self):
        
        if self.params[self.gamma]:
            self.G_expr = (
            self.u_x * ((self.phi_E + self.phi_P) / 2 - self.phi_x_0) + self.u_y * (self.phi_y_L - (self.phi_P + self.phi_S) / 2) - 
            self.gamma * (
                (self.phi_E - self.phi_P) / self.delta_x - (self.phi_P - self.phi_x_0) / (self.delta_x / 2) + 
                (self.phi_y_L - self.phi_P) / (self.delta_y / 2) - (self.phi_P - self.phi_S) / self.delta_y)
            
            )
        else:
            self.G_expr = self.u_x * ((self.phi_E - self.phi_P) / (self.delta_x) - (self.phi_P - self.phi_x_0) / (self.delta_x / 2)) + self.u_y * ((self.phi_y_L - self.phi_P) / (self.delta_y / 2)- (self.phi_P - self.phi_S) / (self.delta_y))
        
        self.G_expr = simplify(self.G_expr.subs(self.params))

        
        
        
        
    def setHExpr(self):
        
        if self.params[self.gamma]:
            self.H_expr = (
            self.u_x * ((self.phi_E + self.phi_P) / 2 - (self.phi_P + self.phi_W) / 2) + self.u_y * (self.phi_y_L - (self.phi_P + self.phi_S) / 2) - 
            self.gamma * (
                (self.phi_E - self.phi_P) / self.delta_x - (self.phi_P - self.phi_W) / self.delta_x + 
                (self.phi_y_L - self.phi_P) / (self.delta_y / 2) - (self.phi_P - self.phi_S) / self.delta_y)
            
            )
        else:
            self.H_expr = self.u_x * ((self.phi_E - self.phi_P) / (self.delta_x) - (self.phi_P - self.phi_W) / (self.delta_x)) + self.u_y * ((self.phi_y_L - self.phi_P) / (self.delta_y / 2)- (self.phi_P - self.phi_S) / (self.delta_y))
        self.H_expr = simplify(self.H_expr.subs(self.params))
        
        
        
        
    def setIExpr(self):
        
        if self.params[self.gamma]:
            self.I_expr = (
            self.u_x * (self.phi_x_L - (self.phi_P + self.phi_W) / 2) + self.u_y * (self.phi_y_L - (self.phi_P + self.phi_S) / 2) - 
            self.gamma * (
                (self.phi_x_L - self.phi_P) / (self.delta_x / 2) - (self.phi_P - self.phi_W) / self.delta_x + 
                (self.phi_y_L - self.phi_P) / (self.delta_y / 2) - (self.phi_P - self.phi_S) / self.delta_y)
            
            )
        else:
            self.I_expr = self.u_x * ((self.phi_x_L - self.phi_P) / (self.delta_x / 2) - (self.phi_P - self.phi_W) / (self.delta_x)) + self.u_y * ((self.phi_y_L - self.phi_P) / (self.delta_y / 2)- (self.phi_P - self.phi_S) / (self.delta_y))
        self.I_expr = simplify(self.I_expr.subs(self.params))