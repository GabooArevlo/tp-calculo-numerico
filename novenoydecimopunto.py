# =========================================================
# PUNTOS 9 Y 10: Análisis de Resultados y Gráficos de Convergencia
# =========================================================
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# --- 1. DEFINICIONES ANALÍTICAS Y PREPARACIÓN ---
x_sym, y_sym = sp.symbols('x y')
r_squared_sym = sp.Rational(1, 6)**2

# Solución analítica u_a
u_d = x_sym * (1 - x_sym) * y_sym * (1 - y_sym)
u_n = ((x_sym - 0.5)**2 + (y_sym - 0.5)**2 - r_squared_sym)**2
u_a = u_d * u_n
lap_u_a = sp.diff(u_a, x_sym, 2) + sp.diff(u_a, y_sym, 2)

f_a_func = sp.lambdify((x_sym, y_sym), sp.simplify(lap_u_a), "numpy")
u_a_func = sp.lambdify((x_sym, y_sym), u_a, "numpy")

def f_xy(xv, yv):
    return f_a_func(xv, yv)

# Dimensiones N (N x N nodos) a analizar
N_list = [9, 11, 21, 31, 41, 51]
results = [] # Lista para almacenar los resultados del Punto 9

# --- 2. REPETICIÓN DEL CÁLCULO DE CONVERGENCIA (Punto 8) ---
print("Realizando cálculos de convergencia para obtener métricas...")

for N in N_list:
    n = N
    L = 1.0
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    h = x[1] - x[0] # Separación entre nodos
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

    # 2.2. Índice y Construcción del Sistema
    idx_map = -np.ones(N_total, dtype=int)
    counter = 0
    for k in range(N_total):
        if node_type[k] in ['Interno', 'Neumann']:
            idx_map[k] = counter
            counter += 1
    N_unknowns = counter

    A = lil_matrix((N_unknowns, N_unknowns), dtype=float)
    b = np.zeros(N_unknowns)

    for i in range(n):
        for j in range(n):
            k = ij_to_k(i, j)
            row = idx_map[k]
            if row == -1: continue

            xk, yk = coords[k]
            center_coeff = 4.0 
            b_val = -h**2 * f_xy(xk, yk) 
            
            neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            for (ii, jj) in neighbors:
                kk = ij_to_k(ii, jj)
                if 0 <= ii < n and 0 <= jj < n:
                    tipo_vecino = node_type[kk]
                    if tipo_vecino == 'Externo':
                        center_coeff -= 1.0 
                    elif tipo_vecino != 'Dirichlet':
                        neighbor_idx = idx_map[kk]
                        A[row, neighbor_idx] = -1.0
            A[row, row] = center_coeff
            b[row] = b_val

    # 2.3. Resolver, Reconstruir y Calcular Error
    A_csr = csr_matrix(A)
    u_vec = spsolve(A_csr, b)

    U_grid = np.full(N_total, np.nan, dtype=float)
    U_grid[is_dirichlet] = 0.0
    for k in range(N_total):
        mk = idx_map[k]
        if mk != -1:
            U_grid[k] = u_vec[mk]
            
    U_grid_2D = U_grid.reshape((n, n))
    U_analytic_2D = u_a_func(X, Y).reshape((n, n))
    
    E_abs_2D = np.abs(U_analytic_2D - U_grid_2D)
    valid_mask = ~np.isnan(E_abs_2D)
    E_abs_valid = E_abs_2D[valid_mask]

    E_MAX = np.max(E_abs_valid)
    E_promedio = np.mean(E_abs_valid)
    
    results.append({
        'N_total': N_total, 
        'h': h, 
        'E_MAX': E_MAX, 
        'E_promedio': E_promedio
    })

# --- 3. PUNTO 9: Tabla de Resultados ---
print("\n" + "="*50)
print("PUNTO 9: TABLA DE CONVERGENCIA (Método de Diferencias Finitas)")
print("="*50)

print("{:^10} | {:^15} | {:^15} | {:^15}".format("Nodos Total", "Separación h", "E_A Máximo", "E_A Promedio"))
print("-" * 60)
for r in results:
    print("{:<10} | {:<15.6e} | {:<15.6e} | {:<15.6e}".format(
        r['N_total'], r['h'], r['E_MAX'], r['E_promedio']
    ))
print("-" * 60)

# --- 4. PUNTO 10: Gráficos de Convergencia 2D ---

N_total_arr = np.array([r['N_total'] for r in results])
E_MAX_arr = np.array([r['E_MAX'] for r in results])
E_promedio_arr = np.array([r['E_promedio'] for r in results])

fig, ax = plt.subplots(figsize=(10, 6))

# Gráfico 1: E_A Máximo vs. Cantidad de Nodos
ax.plot(N_total_arr, E_MAX_arr, 'o-', label='$E_{A, \text{Máximo}}$', color='red')
ax.set_title('Punto 10: Convergencia del Error Máximo vs. Cantidad de Nodos')
ax.set_xlabel('Cantidad de Nodos (N²)')
ax.set_ylabel('Error Absoluto Máximo ($E_{MAX}$)')
ax.grid(True, which="both", ls="--")
ax.legend()
plt.show()

# Gráfico 2: E_A Promedio vs. Cantidad de Nodos
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(N_total_arr, E_promedio_arr, 's-', label='$E_{A, \text{Promedio}}$', color='blue')
ax.set_title('Punto 10: Convergencia del Error Promedio vs. Cantidad de Nodos')
ax.set_xlabel('Cantidad de Nodos (N²)')
ax.set_ylabel('Error Absoluto Promedio ($E_{\text{promedio}}$)')
ax.grid(True, which="both", ls="--")
ax.legend()
plt.show()

print("\nAnálisis DF de los Puntos 9 y 10 completado. ✔")