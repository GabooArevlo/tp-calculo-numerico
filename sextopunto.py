# =========================================================
# PUNTO 6: Resolver A u = b (Sistema del Punto 5)
# - Autosuficiente con f_a(x,y) correcta.
# - Utiliza el estencil modificado para Neumann.
# =========================================================
import numpy as np
import sympy as sp
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve # Solucionador para matrices dispersas
import matplotlib.pyplot as plt

# --- 1. PLANTEO ANALÍTICO Y FUNCIÓN FUENTE (f_a) ---
x_sym, y_sym = sp.symbols('x y')
r_squared_sym = sp.Rational(1, 6)**2

u_d = x_sym * (1 - x_sym) * y_sym * (1 - y_sym)
u_n = ((x_sym - 0.5)**2 + (y_sym - 0.5)**2 - r_squared_sym)**2
u_a = u_d * u_n
lap_u_a = sp.diff(u_a, x_sym, 2) + sp.diff(u_a, y_sym, 2)
f_a_func = sp.lambdify((x_sym, y_sym), sp.simplify(lap_u_a), "numpy")

def f_xy(xv, yv):
    """Función fuente (f_a = Delta u_a) para el RHS del sistema."""
    return f_a_func(xv, yv)

# --- 2. MALLADO Y CLASIFICACIÓN DE NODOS (Igual que Punto 5) ---
n = 7
L = 1.0
x = np.linspace(0, L, n)
y = np.linspace(0, L, n)
h = x[1] - x[0]
N_total = n * n

X, Y = np.meshgrid(x, y)
coords = np.column_stack((X.flatten(), Y.flatten()))

center = (0.5, 0.5)
r = 0.3333 / 2.0 # radio del hueco

# Clasificaciones base
is_dirichlet = (np.isclose(coords[:,0], 0) | np.isclose(coords[:,0], L) |
                np.isclose(coords[:,1], 0) | np.isclose(coords[:,1], L))
dist_to_center = np.sqrt((coords[:,0] - center[0])**2 + (coords[:,1] - center[1])**2)
is_inside_hole = dist_to_center < r - 1e-12

node_type = np.full(N_total, 'Interno', dtype=object)
node_type[is_dirichlet] = 'Dirichlet'
node_type[is_inside_hole] = 'Externo'

def ij_to_k(i, j): return i * n + j

# Detectar nodos Neumann
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

# Crear índice de incógnitas (Solo Internos y Neumann)
idx_map = -np.ones(N_total, dtype=int)
counter = 0
for k in range(N_total):
    if node_type[k] in ['Interno', 'Neumann']:
        idx_map[k] = counter
        counter += 1
N_unknowns = counter

# --- 3. CONSTRUCCIÓN DEL SISTEMA A u = b (Corregida) ---
A = lil_matrix((N_unknowns, N_unknowns), dtype=float)
b = np.zeros(N_unknowns)

for i in range(n):
    for j in range(n):
        k = ij_to_k(i, j)
        row = idx_map[k]
        if row == -1: 
            continue # Nodos Dirichlet y Externos no son incógnitas

        xk, yk = coords[k]
        
        # 4u - sum(u_vecinos) = -h^2 f
        center_coeff = 4.0 
        b_val = -h**2 * f_xy(xk, yk) # RHS usando f_a(x,y)
        
        neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
        
        for (ii, jj) in neighbors:
            kk = ij_to_k(ii, jj)
            
            if 0 <= ii < n and 0 <= jj < n:
                tipo_vecino = node_type[kk]
                
                # Caso 1: Vecino es Externo (Hueco) -> Condición Neumann (u_vecino = u_center)
                if tipo_vecino == 'Externo':
                    center_coeff -= 1.0 # Modifica el coeficiente central: 4 -> 3
                    
                # Caso 2: Vecino es Dirichlet (Frontera Exterior, u=0)
                elif tipo_vecino == 'Dirichlet':
                    # Término del Laplaciano: -1 * u_dirichlet = -1 * 0. No afecta b_val.
                    pass 
                
                # Caso 3: Vecino es otra Incógnita
                else:
                    neighbor_idx = idx_map[kk]
                    A[row, neighbor_idx] = -1.0 # Coeficiente -1 para el Laplaciano

        # Fijar coeficiente diagonal
        A[row, row] = center_coeff
        
        # Fijar RHS
        b[row] = b_val

# --- 4. RESOLVER A u = b ---
print("\nResolviendo el sistema lineal...")
A_csr = csr_matrix(A)
u_vec = spsolve(A_csr, b)

# --- 5. RECONSTRUIR SOLUCIÓN Y GRAFICAR ---
U_grid = np.full(N_total, np.nan, dtype=float)

# Poner Dirichlet = 0 en fronteras exteriores
U_grid[is_dirichlet] = 0.0

# Poner incógnitas resueltas
for k in range(N_total):
    mk = idx_map[k]
    if mk != -1:
        U_grid[k] = u_vec[mk]
        
U_grid_2D = U_grid.reshape((n, n))

print(f"\nSolución u en nodos ({N_unknowns} incógnitas):")
print(U_grid_2D)

# Gráfico de la solución
plt.figure(figsize=(7,7))
plt.imshow(U_grid_2D, origin='lower', extent=(0,1,0,1))
plt.colorbar(label='u (aprox)')

# Marcar el hueco (círculo interior)
center_p = (0.5, 0.5)
circle = plt.Circle(center_p, r, color='white', fill=True, zorder=10)
plt.gca().add_patch(circle)

plt.title(f"Solución DF (n={n}, f=$f_a$)")
plt.xlabel('x'); plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.show()