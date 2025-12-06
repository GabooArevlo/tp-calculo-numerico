import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import sympy as sp

# ==========================================
# 1. PLANTEO ANALÍTICO Y FUNCIÓN FUENTE (f_a)
# ==========================================
print("1. Calculando la función fuente analítica f_a(x,y) = Delta u_a...")
x_sym, y_sym = sp.symbols('x y')
r_squared = sp.Rational(1, 6)**2

# Solución de Dirichlet (u=0 en el cuadrado)
u_d = x_sym * (1 - x_sym) * y_sym * (1 - y_sym)
# Función de anulación (u_n=0 en el círculo, y derivada normal de u_n=0 en el círculo)
u_n = ((x_sym - 0.5)**2 + (y_sym - 0.5)**2 - r_squared)**2

# Solución analítica de referencia (CORREGIDA)
u_a = u_d * u_n

# Calcular la función fuente f_a(x,y) = Delta u_a
lap_u_a = sp.diff(u_a, x_sym, 2) + sp.diff(u_a, y_sym, 2)
lap_u_a_simpl = sp.simplify(lap_u_a)

# Convertir la función fuente simbólica a una función NumPy para evaluación rápida
f_a_func = sp.lambdify((x_sym, y_sym), lap_u_a_simpl, "numpy")

def f_xy(xv, yv):
    """Función fuente que se evaluará en los puntos de la malla."""
    # Usamos la función fuente analítica (f_a) para que el problema sea consistente
    return f_a_func(xv, yv)

# ==========================================
# 2. MALLA Y CLASIFICACIÓN DE NODOS
# ==========================================
print("2. Clasificando nodos...")
n = 7                  # 7x7 = 49 puntos
L = 1.0
x = np.linspace(0, L, n)
y = np.linspace(0, L, n)
h = x[1] - x[0]        # Paso de malla (1/6)

center = (0.5, 0.5)
r = 0.3333 / 2.0       # Radio del hueco (aprox 1/6)

X, Y = np.meshgrid(x, y)
coords = np.column_stack((X.flatten(), Y.flatten()))
N_total = n * n

# Clasificación
is_dirichlet = (np.isclose(coords[:,0], 0) | np.isclose(coords[:,0], L) |
                np.isclose(coords[:,1], 0) | np.isclose(coords[:,1], L))

dist_to_center = np.sqrt((coords[:,0] - center[0])**2 + (coords[:,1] - center[1])**2)
# Usamos r para la exclusión del nodo, si r=1/6, solo el nodo central (0.5, 0.5) caerá dentro.
is_inside_hole = dist_to_center < r - 1e-12 

# Mapeo de índices de nodos (0, 1, 2, ... N_unknowns-1)
idx_map = -np.ones(N_total, dtype=int)
idx = 0
for k in range(N_total):
    if (not is_dirichlet[k]) and (not is_inside_hole[k]):
        idx_map[k] = idx
        idx += 1
N_unknowns = idx

print(f"   Nodos totales: {N_total}")
print(f"   Nodos Dirichlet (exteriores): {is_dirichlet.sum()}")
print(f"   Nodos dentro del hueco (excluidos): {is_inside_hole.sum()}")
print(f"   Nodos incógnita (efectivos): {N_unknowns}")
print(f"   h (paso de malla): {h}")

# ==========================================
# 3. CONSTRUCCIÓN DEL SISTEMA A u = b
# ==========================================
print("3. Construyendo la matriz A y el vector b...")
A = lil_matrix((N_unknowns, N_unknowns), dtype=float)
b = np.zeros(N_unknowns, dtype=float)

# Helper para convertir (i,j) a índice lineal k
def ij_to_k(i, j):
    return i * n + j

# Recorremos todos los nodos
for i in range(n):
    for j in range(n):
        k = ij_to_k(i, j)
        unk = idx_map[k]
        if unk == -1:
            continue  # No es una incógnita (es frontera o hueco)

        px, py = coords[k]
        # RHS = f(x,y) * h^2
        b_val = f_xy(px, py) * (h**2)

        # Estencil de 5 puntos: vecinos
        center_coeff = -4.0 # Coeficiente central inicial (-4 para Laplaciano estándar)

        neighbors = [ (i+1, j), (i-1, j), (i, j+1), (i, j-1) ]
        for (ii, jj) in neighbors:
            kk = ij_to_k(ii, jj)
            
            # Condición de Neumann (aproximación simple: u_vecino = u_center)
            if is_inside_hole[kk]:
                # Vecino está en el hueco (Γn) -> ∂u/∂n = 0 se aproxima con u_vecino = u_center
                # Esto mueve +1 de A_vecino a A_central, cambiando A[unk, unk] de -4 a -3, -2, etc.
                center_coeff += 1.0 
                
            # Condición de Dirichlet (u=0)
            elif is_dirichlet[kk]:
                # Vecino es frontera exterior con u=0 -> El término se mueve al RHS, pero es 0.
                pass 
                
            # Vecino es otra incógnita
            else:
                neighbor_idx = idx_map[kk]
                A[unk, neighbor_idx] = 1.0 # Coeficiente 1 para el vecino

        # Fijar coeficiente diagonal
        A[unk, unk] = center_coeff

        # Fijar RHS
        b[unk] = b_val

# ==========================================
# 4. RESOLVER Y RECONSTRUIR
# ==========================================
print("4. Resolviendo el sistema lineal...")
A_csr = csr_matrix(A)
u_vec = spsolve(A_csr, b) # Resuelve A u = b

# Reconstruir solución en la malla completa (NaN dentro del hueco para graficar)
U_grid = np.full(N_total, np.nan, dtype=float)

# 1. Poner Dirichlet = 0 en fronteras exteriores
U_grid[is_dirichlet] = 0.0
# 2. Poner incógnitas resueltas
for k in range(N_total):
    mk = idx_map[k]
    if mk != -1:
        U_grid[k] = u_vec[mk]
        
# 3. Reshape para graficar en 2D
U_grid = U_grid.reshape((n, n))

# ==========================================
# 5. GRAFICAR RESULTADO
# ==========================================
plt.figure(figsize=(7,7))

# Graficar la solución con imshow
plt.imshow(U_grid, origin='lower', extent=(0,1,0,1))
cbar = plt.colorbar(label='$u_{aprox}$')

# Marcar el hueco (círculo interior)
center_p = (0.5, 0.5)
circle = plt.Circle(center_p, r, color='white', fill=True, zorder=10)
plt.gca().add_patch(circle)

# Marcar los nodos incógnita (puntos negros)
plt.scatter(coords[~is_dirichlet & ~is_inside_hole][:,0],
            coords[~is_dirichlet & ~is_inside_hole][:,1],
            c='k', s=20, zorder=11, label='Nodos Incógnita')

plt.title("Solución aproximada por Diferencias Finitas (n=7)")
plt.xlabel('x'); plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.grid(False)
plt.show()

print("\nProceso de DF finalizado. La solución usa la f_a(x,y) correcta.")