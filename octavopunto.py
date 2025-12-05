# =========================================================
# PUNTO 8: Estudio de Convergencia por Refinamiento de Malla
# =========================================================
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D

# --- 1. DEFINICIONES ANALÍTICAS ---
x_sym, y_sym = sp.symbols('x y')
r_squared_sym = sp.Rational(1, 6)**2

# Solución analítica u_a
u_d = x_sym * (1 - x_sym) * y_sym * (1 - y_sym)
u_n = ((x_sym - 0.5)**2 + (y_sym - 0.5)**2 - r_squared_sym)**2
u_a = u_d * u_n

# Función Fuente f_a (Laplaciano de u_a)
lap_u_a = sp.diff(u_a, x_sym, 2) + sp.diff(u_a, y_sym, 2)
f_a_func = sp.lambdify((x_sym, y_sym), sp.simplify(lap_u_a), "numpy")
u_a_func = sp.lambdify((x_sym, y_sym), u_a, "numpy")

def f_xy(xv, yv):
    return f_a_func(xv, yv)

# Lista de dimensiones N (N x N nodos)
N_list = [9, 11, 21, 31, 41, 51]
N_total_list = [N**2 for N in N_list]

results = [] # Para almacenar E_MAX y E_promedio

# --- 2. BUCLE DE CONVERGENCIA ---
print("Iniciando estudio de convergencia...")
for N in N_list:
    n = N
    L = 1.0
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    h = x[1] - x[0]
    N_total = n * n
    
    X, Y = np.meshgrid(x, y)
    coords = np.column_stack((X.flatten(), Y.flatten()))

    center = (0.5, 0.5)
    r = 0.3333 / 2.0 
    
    # 2.1. Clasificación de Nodos
    is_dirichlet = (np.isclose(coords[:,0], 0) | np.isclose(coords[:,0], L) |
                    np.isclose(coords[:,1], 0) | np.isclose(coords[:,1], L))
    dist_to_center = np.sqrt((coords[:,0] - center[0])**2 + (coords[:,1] - center[1])**2)
    is_inside_hole = dist_to_center < r - 1e-12

    node_type = np.full(N_total, 'Interno', dtype=object)
    node_type[is_dirichlet] = 'Dirichlet'
    node_type[is_inside_hole] = 'Externo'

    def ij_to_k(i, j): return i * n + j

    # Detectar nodos Neumann (vecinos a 'Externo')
    for i in range(n):
        for j in range(n):
            k = ij_to_k(i, j)
            if node_type[k] != 'Interno': continue
            for ii, jj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if 0 <= ii < n and 0 <= jj < n:
                    kk = ij_to_k(ii, jj)
                    if node_type[kk] == 'Externo':
                        node_type[k] = 'Neumann'
                        break

    # 2.2. Índice de Incógnitas
    idx_map = -np.ones(N_total, dtype=int)
    counter = 0
    for k in range(N_total):
        if node_type[k] in ['Interno', 'Neumann']:
            idx_map[k] = counter
            counter += 1
    N_unknowns = counter

    # 2.3. Construcción del Sistema A u = b
    A = lil_matrix((N_unknowns, N_unknowns), dtype=float)
    b = np.zeros(N_unknowns)

    for i in range(n):
        for j in range(n):
            k = ij_to_k(i, j)
            row = idx_map[k]
            if row == -1: continue

            xk, yk = coords[k]
            
            # 4u - sum(u_vecinos) = -h^2 f
            center_coeff = 4.0 
            b_val = -h**2 * f_xy(xk, yk) 
            
            neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            
            for (ii, jj) in neighbors:
                kk = ij_to_k(ii, jj)
                if 0 <= ii < n and 0 <= jj < n:
                    tipo_vecino = node_type[kk]
                    
                    if tipo_vecino == 'Externo':
                        # Neumann (Estencil Modificado): u_vecino = u_center
                        center_coeff -= 1.0 
                    elif tipo_vecino == 'Dirichlet':
                        # Dirichlet (u=0): El término se omite del LHS y no afecta RHS.
                        pass 
                    else: # Vecino es otra Incógnita
                        neighbor_idx = idx_map[kk]
                        A[row, neighbor_idx] = -1.0

            A[row, row] = center_coeff
            b[row] = b_val

    # 2.4. Resolver A u = b
    A_csr = csr_matrix(A)
    u_vec = spsolve(A_csr, b)

    # 2.5. Reconstruir Solución
    U_grid = np.full(N_total, np.nan, dtype=float)
    U_grid[is_dirichlet] = 0.0

    for k in range(N_total):
        mk = idx_map[k]
        if mk != -1:
            U_grid[k] = u_vec[mk]
            
    U_grid_2D = U_grid.reshape((n, n))
    
    # 2.6. Calcular Error
    U_analytic_2D = u_a_func(X, Y).reshape((n, n))
    
    E_abs_2D = np.abs(U_analytic_2D - U_grid_2D)
    valid_mask = ~np.isnan(E_abs_2D)
    E_abs_valid = E_abs_2D[valid_mask]

    E_MAX = np.max(E_abs_valid)
    E_promedio = np.mean(E_abs_valid)
    
    results.append({'N_total': N_total, 'N': N, 'E_MAX': E_MAX, 'E_promedio': E_promedio, 'U_DF': U_grid_2D, 'U_A': U_analytic_2D, 'E_A': E_abs_2D})
    
    print(f"Malla {N}x{N} ({N_total} nodos): E_MAX = {E_MAX:.6e}")
    
# --- 3. GRÁFICOS (Para el último N o el requerido) ---

def plot_3d(X, Y, Z, title, zlabel, N_nodes, E_max=None):
    """Función auxiliar para graficar superficies 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    subtitle = f"(N={N_nodes} nodos)"
    if E_max is not None:
         subtitle += f"\nE_MAX: {E_max:.2e}"
         
    ax.set_title(f'{title} {subtitle}')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel(zlabel)
    plt.show()

# Graficar los 3D para la última malla (N=51)
last_res = results[-1]
X_last, Y_last = np.meshgrid(np.linspace(0, L, last_res['N']), np.linspace(0, L, last_res['N']))

# Gráfico 3D de la Solución u_aprox (DF)
plot_3d(X_last, Y_last, last_res['U_DF'], "Solución DF Aproximada", "u (DF)", last_res['N_total'])

# Gráfico 3D de la Solución u_a (Analítica)
plot_3d(X_last, Y_last, last_res['U_A'], "Solución Analítica $u_a$", "u (Analítica)", last_res['N_total'])

# Gráfico 3D del Error Absoluto E_A
plot_3d(X_last, Y_last, last_res['E_A'], "Error Absoluto $|u_a - u_{DF}|$", "Error Absoluto $E_A$", last_res['N_total'], last_res['E_MAX'])

print("\nResultados de las métricas (Para usar en el Punto 10):")
print("{:^10} | {:^10} | {:^15} | {:^15}".format("N Total", "N x N", "E_MAX", "E_Promedio"))
print("-" * 55)
for r in results:
    print("{:<10} | {:<10} | {:<15.6e} | {:<15.6e}".format(r['N_total'], r['N'], r['E_MAX'], r['E_promedio']))

print("\nFIN DEL CÓDIGO DE CONVERGENCIA ✔")