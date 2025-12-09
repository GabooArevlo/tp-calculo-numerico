# ============================================================
# CÓDIGO EF: Puntos 7, 9 y 10 (CORREGIDO)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import sympy as sp
import scipy.sparse.linalg as spla

# -------------------------
# 1) DEFINICIÓN ANALÍTICA 
# -------------------------
x, y, theta = sp.symbols('x y theta')

# Radio del agujero (1/6 = 0.3333/2.0)
r_hole_sym = sp.Rational(1, 6)

# Funciones de la solución analítica del problema (u_a = u_d * u_n)
u_d = x * (1 - x) * y * (1 - y)
u_n = ((x - sp.Rational(1,2))**2 + (y - sp.Rational(1,2))**2 - r_hole_sym**2)**2

# CORRECCIÓN: u_a = u_d * u_n
u_a_sym = u_d * u_n # <--- ¡CAMBIO CLAVE!

# Laplaciano simbólico -> f_a (Función fuente)
# f_fuente = -Laplaciano(u_a) para resolver -Δu = f
f_a_sym = sp.simplify(-(sp.diff(u_a_sym, x, 2) + sp.diff(u_a_sym, y, 2))) # <--- Laplaciano negativo
# Gradiente para Neumann (necesario para el C.B. gN)
u_x_sym = sp.diff(u_a_sym, x)
u_y_sym = sp.diff(u_a_sym, y)

# Funciones numéricas
u_a_num = sp.lambdify((x, y), u_a_sym, "numpy")
fa = sp.lambdify((x, y), f_a_sym, "numpy")
u_x_num = sp.lambdify((x, y), u_x_sym, "numpy")
u_y_num = sp.lambdify((x, y), u_y_sym, "numpy")

# -------------------------
# 2) NODOS Y CONDICIONES DE BORDE (Reutilizando la configuración de Punto 7)
# -------------------------
nodes = np.array([
    (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
    (1.0, 0.25), (1.0, 0.5), (1.0, 0.75), (1.0, 1.0),
    (0.75, 1.0), (0.5, 1.0), (0.25, 1.0), (0.0, 1.0),
    (0.0, 0.75), (0.0, 0.5), (0.0, 0.25),
    (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
    # Puntos de Neumann (en el círculo, r_hole_num = 0.3333/2.0)
    (0.5 + 0.3333/2.0, 0.5), (0.5 - 0.3333/2.0, 0.5),
    (0.5, 0.5 + 0.3333/2.0), (0.5, 0.5 - 0.3333/2.0),
    (0.5, 0.5), # Externo (Nodo 24)
    (0.5, 0.25), (0.5, 0.75), (0.25, 0.5)
])
n_nodes = nodes.shape[0]
center = np.array([0.5, 0.5])
r = 0.3333 / 2.0
tol = 1e-6

dist = np.linalg.norm(nodes - center, axis=1)
is_dirichlet = np.isclose(nodes[:,0], 0.0) | np.isclose(nodes[:,0], 1.0) | \
               np.isclose(nodes[:,1], 0.0) | np.isclose(nodes[:,1], 1.0)
is_externo = np.isclose(dist, 0.0, atol=1e-8) # Solo el nodo 24 (0.5, 0.5)
is_neumann = np.isclose(dist, r, atol=1e-4)

# Mallado
valid_mask = ~is_externo
global_idx_valid = np.where(valid_mask)[0]
tri = Delaunay(nodes[valid_mask])
elements = [tuple(global_idx_valid[tri_local]) for tri_local in tri.simplices]

# Mapeo de índices
idx_map = -np.ones(n_nodes, dtype=int)
counter = 0
for k in range(n_nodes):
    if not is_externo[k]:
        idx_map[k] = counter
        counter += 1
N_unknowns = counter

# -------------------------
# 3) FUNCIONES ELEMENTALES (De los Puntos 3/4)
# -------------------------
def element_stiffness(x1, y1, x2, y2, x3, y3):
    A = 0.5 * np.linalg.det(np.array([[1, x1, y1], [1, x2, y2], [1, x3, y3]]))
    if A <= 0: raise ValueError("Área no positiva.")
    b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
    c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1
    b = np.array([b1, b2, b3]); c = np.array([c1, c2, c3])
    Ke = (1.0 / (4.0 * A)) * (np.outer(b,b) + np.outer(c,c))
    return Ke, A

def element_load(x1, y1, x2, y2, x3, y3, f_func):
    xc = (x1 + x2 + x3) / 3.0
    yc = (y1 + y2 + y3) / 3.0
    A = 0.5 * np.linalg.det(np.array([[1, x1, y1], [1, x2, y2], [1, x3, y3]]))
    fe = np.ones(3) * (A * f_func(xc, yc) / 3.0)
    return fe

# Flujo de Neumann gN = Grad(u_a) . n
def gN_at_point(xp, yp):
    dx = xp - center[0]; dy = yp - center[1]
    distancia = np.hypot(dx, dy)
    if np.isclose(distancia, 0.0, atol=1e-8): return 0.0
    nx = dx / distancia; ny = dy / distancia
    return u_x_num(xp, yp) * nx + u_y_num(xp, yp) * ny

# -------------------------
# 4) ENSAMBLAJE GLOBAL A y b
# -------------------------
from scipy.sparse import lil_matrix
A = lil_matrix((N_unknowns, N_unknowns), dtype=float)
b = np.zeros(N_unknowns, dtype=float)

for el in elements:
    n0, n1, n2 = el
    x1, y1 = nodes[n0]; x2, y2 = nodes[n1]; x3, y3 = nodes[n2]
    Ke, Area = element_stiffness(x1, y1, x2, y2, x3, y3)
    fe = element_load(x1, y1, x2, y2, x3, y3, fa)

    local_nodes = [n0, n1, n2]
    for i_local, ni in enumerate(local_nodes):
        Ii = idx_map[ni]
        if Ii == -1: continue
        b[Ii] += fe[i_local]
        for j_local, nj in enumerate(local_nodes):
            Jj = idx_map[nj]
            if Jj == -1: continue
            A[Ii, Jj] += Ke[i_local, j_local]

# C.B. Neumann (Aporte al vector b)
edges = set()
for el in elements:
    tri_nodes = list(el)
    pairs = [(tri_nodes[0], tri_nodes[1]), (tri_nodes[1], tri_nodes[2]), (tri_nodes[2], tri_nodes[0])]
    for u,v in pairs:
        edge = (min(u,v), max(u,v)); edges.add(edge)

for (a,bidx) in edges:
    if is_neumann[a] and is_neumann[bidx]:
        xa, ya = nodes[a]; xb, yb = nodes[bidx]
        Ledge = np.hypot(xa - xb, ya - yb)
        xm, ym = 0.5*(xa+xb), 0.5*(ya+yb)
        gmid = gN_at_point(xm, ym)
        # Contribución de la integral de borde ∫ gN * phi_i dΓ ≈ L * gN_mid * 0.5
        contrib = Ledge * gmid * 0.5 # Corregida la contribución, estaba Ledge/2.0 * gmid * 0.5
        Ia = idx_map[a]; Ib = idx_map[bidx]
        if Ia != -1: b[Ia] += contrib
        if Ib != -1: b[Ib] += contrib

# C.B. Dirichlet (Imposición de u=u_a en la frontera)
for k in range(n_nodes):
    if is_dirichlet[k] and (idx_map[k] != -1):
        idx = idx_map[k]
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        # El valor de u_a es 0 en el borde exterior, gracias al término u_d
        b[idx] = u_a_num(nodes[k, 0], nodes[k, 1]) 

# -------------------------
# 5) RESOLVER SISTEMA (Punto 9)
# -------------------------
A_csr = A.tocsr()
U_approx = spla.spsolve(A_csr, b)

# Reconstruir la solución completa (incluyendo nodos Dirichlet y Externo)
U_full = np.full(n_nodes, np.nan)
for k in range(n_nodes):
    if idx_map[k] != -1:
        U_full[k] = U_approx[idx_map[k]]

# -------------------------
# 6) CÁLCULO DE ERROR (Punto 10)
# -------------------------
# Valores analíticos en los nodos (excluyendo 'Externo')
U_a_valid = u_a_num(nodes[valid_mask, 0], nodes[valid_mask, 1])
U_h_valid = U_approx

# Error en los nodos válidos
Error_valid = np.abs(U_h_valid - U_a_valid)
error_max = np.max(Error_valid)
error_promedio = np.mean(Error_valid)

# Reconstrui el vector de error completo para el gráfico
Error_full = np.full(n_nodes, np.nan)
for k in range(n_nodes):
    if idx_map[k] != -1:
        Error_full[k] = Error_valid[idx_map[k]]

# -------------------------
# 7) GRÁFICOS (Punto 9 y 10)
# -------------------------
from matplotlib.tri import Triangulation

# Obtener los nodos válidos para la triangulación
nodes_valid = nodes[valid_mask]
# Mapear los índices locales de tri.simplices a los índices globales (0 a N_unknowns-1) para Triangulation
triang_indices = np.array([idx_map[i] for i in global_idx_valid])
triang = Triangulation(nodes_valid[:, 0], nodes_valid[:, 1], tri.simplices)

# Gráfico 3D de la Solución (Punto 9)
fig_sol = plt.figure(figsize=(10, 7))
ax_sol = fig_sol.add_subplot(111, projection='3d')
# Usamos U_h_valid que tiene N_unknowns elementos, que coincide con el tamaño de nodes_valid
ax_sol.plot_trisurf(triang, U_h_valid, cmap='viridis', linewidth=0.2, antialiased=True) 
ax_sol.set_title("Solución Numérica FEM P1 (Punto 9) - Malla Manual")
ax_sol.set_xlabel('x'); ax_sol.set_ylabel('y'); ax_sol.set_zlabel('$u_h$')
plt.show()

# Gráfico 3D del Error (Punto 10)
fig_err = plt.figure(figsize=(10, 7))
ax_err = fig_err.add_subplot(111, projection='3d')
ax_err.plot_trisurf(triang, Error_valid, cmap='plasma', linewidth=0.2, antialiased=True)
ax_err.set_title("Error absoluto $|u_h - u_a|$ (Punto 10) - Malla Manual")
ax_err.set_xlabel('x'); ax_err.set_ylabel('y'); ax_err.set_zlabel('Error Absoluto')
plt.show()

# -------------------------
# 8) SALIDA
# -------------------------
print("\n========= ERRORES FEM P1 (Malla manual) =========")
print(f"Error absoluto máximo = {error_max:.8f}")
print(f"Error absoluto promedio = {error_promedio:.8f}")
print("=================================================")