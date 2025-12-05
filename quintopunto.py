# =========================================================
# PUNTO 5 DF CORREGIDO: Construcción de A u = b
# - Usa la función fuente f_a(x,y) correcta (Laplaciano de u_a).
# - Implementa Neumann con Estencil Modificado (Primer Orden).
# =========================================================
import numpy as np
import pandas as pd
import sympy as sp
from scipy.sparse import lil_matrix, csr_matrix

# --- 1. PLANTEO ANALÍTICO Y FUNCIÓN FUENTE (f_a) ---
print("1. Calculando la función fuente analítica f_a(x,y)...")
x_sym, y_sym = sp.symbols('x y')
r_squared = sp.Rational(1, 6)**2

u_d = x_sym * (1 - x_sym) * y_sym * (1 - y_sym)
u_n = ((x_sym - 0.5)**2 + (y_sym - 0.5)**2 - r_squared)**2
u_a = u_d * u_n
lap_u_a = sp.diff(u_a, x_sym, 2) + sp.diff(u_a, y_sym, 2)
f_a_func = sp.lambdify((x_sym, y_sym), sp.simplify(lap_u_a), "numpy")

def f_xy(xv, yv):
    """Función fuente que se evaluará en los puntos de la malla (f_a = Delta u_a)."""
    return f_a_func(xv, yv)


# --- 2. MALLADO Y CLASIFICACIÓN DE NODOS ---
print("2. Clasificando nodos...")
n = 7
L = 1.0
x = np.linspace(0, L, n)
y = np.linspace(0, L, n)
h = x[1] - x[0]
N_total = n * n

X, Y = np.meshgrid(x, y)
coords = np.column_stack((X.flatten(), Y.flatten()))

center = (0.5, 0.5)
r = 0.3333 / 2.0

# Clasificaciones base
is_dirichlet = (np.isclose(coords[:,0], 0) | np.isclose(coords[:,0], L) |
                np.isclose(coords[:,1], 0) | np.isclose(coords[:,1], L))

dist_to_center = np.sqrt((coords[:,0] - center[0])**2 + (coords[:,1] - center[1])**2)
is_inside_hole = dist_to_center < r - 1e-12

node_type = np.full(N_total, 'Interno', dtype=object)
node_type[is_dirichlet] = 'Dirichlet'
node_type[is_inside_hole] = 'Externo'

# Detectar nodos Neumann (vecinos al hueco)
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

# Crear índice de incógnitas (Interno y Neumann son incógnitas)
idx_map = -np.ones(N_total, dtype=int)
counter = 0
for k in range(N_total):
    if node_type[k] in ['Interno', 'Neumann']:
        idx_map[k] = counter
        counter += 1
N_unknowns = counter

# --- 3. CONSTRUCCIÓN DEL SISTEMA A u = b ---
print(f"3. Construyendo la matriz A ({N_unknowns}x{N_unknowns}) y el vector b...")

# Usamos lil_matrix para un llenado eficiente de la matriz dispersa
A = lil_matrix((N_unknowns, N_unknowns), dtype=float)
b = np.zeros(N_unknowns)

# Recorremos solo los nodos que son incógnitas
for i in range(n):
    for j in range(n):
        k = ij_to_k(i, j)
        row = idx_map[k]
        if row == -1: 
            continue

        xk, yk = coords[k]
        tipo = node_type[k]
        
        # Coeficiente central inicial del Laplaciano: -4
        center_coeff = -4.0 
        
        # 3.1. Nodos Dirichlet (No son incógnita en esta versión, su valor es 0)
        # NOTA: Los nodos Dirichlet (idx=-1) no se incluyen en la matriz A. 
        # Si queremos incluirlos, se haría como en tu código original (A[row,row]=1, b[row]=0),
        # pero es más eficiente excluirlos y fijar el valor u=0 después de la solución.

        # 3.2. Nodos Internos y Neumann (Ambos usan estencil modificado)
        
        # Laplaciano: u_{i+1} + u_{i-1} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j} = h^2 f_{i,j}
        # Lo implementamos como: 4u_{i,j} - (suma de vecinos) = -h^2 f_{i,j}

        b_val = -h**2 * f_xy(xk, yk)
        
        neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
        
        # Llenado de vecinos
        for (ii, jj) in neighbors:
            kk = ij_to_k(ii, jj)
            
            if 0 <= ii < n and 0 <= jj < n: # Vecino dentro de la malla
                
                # Caso 1: Vecino es Externo (Hueco) -> Condición Neumann (∂u/∂n=0)
                if node_type[kk] == 'Externo':
                    # Aplicamos Neumann simple: u_vecino = u_center
                    # Esto modifica el coeficiente central: -4 -> -3
                    center_coeff += 1.0 
                    
                # Caso 2: Vecino es Dirichlet (Frontera Exterior, u=0)
                elif node_type[kk] == 'Dirichlet':
                    # El término del vecino u_dirichlet = 0 pasa al RHS, pero como es 0, b no cambia.
                    # El coeficiente de la incógnita del vecino no se añade.
                    pass 
                
                # Caso 3: Vecino es otra Incógnita (Interno o Neumann)
                else:
                    neighbor_idx = idx_map[kk]
                    # Coeficiente -1 (para la forma 4u - sum(u_vecinos))
                    A[row, neighbor_idx] = -1.0 

            # NOTA: El caso del vecino fuera del dominio rectangular no ocurre aquí.

        # Fijar coeficiente diagonal (siempre positivo en esta formulación)
        # El coeficiente central es 4 + (número de vecinos hueco * -1)
        A[row, row] = -center_coeff # Pasamos el -4 (o -3, -2) a positivo 4 (o 3, 2)
        
        # Fijar RHS
        b[row] = b_val


# Muestra de la matriz (como tu original)
A_dense = A.todense()
print("\n===== MATRIZ A (ejemplo, solo muestra el subconjunto de incógnitas) =====")
# Usamos una submatriz representativa para no imprimir toda la matriz 24x24
print(A_dense[:5, :5]) 

print("\n===== VECTOR b (primeros 5 valores, usa f_a(x,y)) =====")
print(b[:5])

print(f"\nNúmero de nodos incógnita (tamaño de A y b): {N_unknowns}")
print("\nFIN DEL CÓDIGO DE CONSTRUCCIÓN ✔")