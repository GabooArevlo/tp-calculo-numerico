# punto7_ensamblado.py
import numpy as np
from scipy.spatial import Delaunay
import sympy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

# -------------------------
# 1) DEFINICIÓN ANALÍTICA
# -------------------------
# Definí aquí ua(x,y). Si querés usar otra, reemplazá la expresión simbólica.
x, y, theta = sp.symbols('x y theta')

# sugerencia (reemplazá si tu ua es otra):
u_d = x*(1 - x)*y*(1 - y)
# u_n como en los mensajes previos (si tu u_n es distinta, cambiála)
u_n = ((x - sp.Rational(1,2))**2 + (y - sp.Rational(1,2))**2 - (sp.Rational(1,6))**2)**2

u_a_sym = u_d - u_n

# Laplaciano simbólico -> f_a
f_a_sym = sp.simplify(sp.diff(u_a_sym, x, 2) + sp.diff(u_a_sym, y, 2))
# grad u for Neumann
u_x_sym = sp.diff(u_a_sym, x)
u_y_sym = sp.diff(u_a_sym, y)

# lambdify numeric functions
fa = sp.lambdify((x, y), f_a_sym, "numpy")
u_x_num = sp.lambdify((x, y), u_x_sym, "numpy")
u_y_num = sp.lambdify((x, y), u_y_sym, "numpy")

# -------------------------
# 2) NODOS (Punto 1: los 28 nodos)
# -------------------------
nodes = np.array([
    (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
    (1.0, 0.25), (1.0, 0.5), (1.0, 0.75), (1.0, 1.0),
    (0.75, 1.0), (0.5, 1.0), (0.25, 1.0), (0.0, 1.0),
    (0.0, 0.75), (0.0, 0.5), (0.0, 0.25),
    (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
    (0.5 + 0.3333/2.0, 0.5), (0.5 - 0.3333/2.0, 0.5),
    (0.5, 0.5 + 0.3333/2.0), (0.5, 0.5 - 0.3333/2.0),
    (0.5, 0.5), (0.5, 0.25), (0.5, 0.75), (0.25, 0.5)
])
n_nodes = nodes.shape[0]

center = np.array([0.5, 0.5])
r = 0.3333 / 2.0
tol = 1e-6

dist = np.linalg.norm(nodes - center, axis=1)
is_dirichlet = np.isclose(nodes[:,0], 0.0) | np.isclose(nodes[:,0], 1.0) | \
               np.isclose(nodes[:,1], 0.0) | np.isclose(nodes[:,1], 1.0)
is_externo = dist < r - tol
is_neumann = np.isclose(dist, r, atol=1e-4)

# valid nodes for triangulation (exclude externo)
valid_mask = ~is_externo
coords_valid = nodes[valid_mask]
global_idx_valid = np.where(valid_mask)[0]

# -------------------------
# 3) TRIANGULACIÓN Delaunay (elementos)
# -------------------------
tri = Delaunay(coords_valid)
elements_local = tri.simplices  # indices locales in coords_valid
elements = [tuple(global_idx_valid[tri_local]) for tri_local in elements_local]

# -------------------------
# 4) funciones elementales
# -------------------------
def element_stiffness(x1, y1, x2, y2, x3, y3):
    A = 0.5 * np.linalg.det(np.array([[1, x1, y1],
                                     [1, x2, y2],
                                     [1, x3, y3]]))
    if A <= 0:
        raise ValueError("Área no positiva para elemento.")
    b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
    c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1
    b = np.array([b1, b2, b3]); c = np.array([c1, c2, c3])
    Ke = (1.0 / (4.0 * A)) * (np.outer(b,b) + np.outer(c,c))
    return Ke, A

def element_load(x1, y1, x2, y2, x3, y3, f_func):
    # centroid quadrature: fe_i = (A/3) * f(xc,yc)
    xc = (x1 + x2 + x3) / 3.0
    yc = (y1 + y2 + y3) / 3.0
    A = 0.5 * np.linalg.det(np.array([[1, x1, y1],
                                     [1, x2, y2],
                                     [1, x3, y3]]))
    fe = np.ones(3) * (A * f_func(xc, yc) / 3.0)
    return fe

# -------------------------
# 5) ENSAMBLAJE GLOBAL A y b
# -------------------------
# mapping global node -> index in unknown system (exclude 'Externo' nodes)
idx_map = -np.ones(n_nodes, dtype=int)
counter = 0
for k in range(n_nodes):
    if not is_externo[k]:
        idx_map[k] = counter
        counter += 1
N_unknowns = counter

# use sparse construction
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

# -------------------------
# 6) APORTES DE NEUMANN EN ARISTAS DEL HUECO (si g != 0)
# -------------------------
# compute derivative normal g_N on the circle using symbolic gradient lambdified
def gN_at_point(xp, yp):
    # normal is (xp - xc, yp - yc)/r
    nx = (xp - center[0]) / np.hypot(xp - center[0], yp - center[1])
    ny = (yp - center[1]) / np.hypot(xp - center[0], yp - center[1])
    return u_x_num(xp, yp)*nx + u_y_num(xp, yp)*ny

# find edges (unique sorted pairs) from elements
edges = set()
for el in elements:
    a, bidx, c = el
    pairs = [(a,bidx),(bidx,c),(c,a)]
    for u,v in pairs:
        edge = (min(u,v), max(u,v))
        edges.add(edge)

# For each edge with both nodes flagged as is_neumann add edge contribution
for (a,bidx) in edges:
    if is_neumann[a] and is_neumann[bidx]:
        xa, ya = nodes[a]; xb, yb = nodes[bidx]
        Ledge = np.hypot(xa - xb, ya - yb)
        xm, ym = 0.5*(xa+xb), 0.5*(ya+yb)
        gmid = gN_at_point(xm, ym)
        # linear basis on edge -> contribute (L/6) * [2*g(a)+g(b)] etc.
        # simplest: distribute integral of g*v over edge equally to the two nodes:
        contrib = (Ledge/2.0) * gmid * 0.5
        Ia = idx_map[a]; Ib = idx_map[bidx]
        if Ia != -1: b[Ia] += contrib
        if Ib != -1: b[Ib] += contrib

# -------------------------
# 7) IMPONER DIRICHLET (bordes exteriores u=0)
# -------------------------
# set rows/cols for Dirichlet nodes
for k in range(n_nodes):
    if is_dirichlet[k] and (idx_map[k] != -1):
        idx = idx_map[k]
        # zero row, set diag=1 and rhs = 0
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        b[idx] = 0.0

# Convert A to CSR for nicer printing/usage
A_csr = A.tocsr()

# -------------------------
# 8) OUTPUT / INFORME
# -------------------------
print("Número de nodos totales:", n_nodes)
print("Número de incógnitas (nodos no 'Externo'):", N_unknowns)
print("Número de elementos (triángulos):", len(elements))
print("Matriz A: shape =", A_csr.shape)
print("Vector b: shape =", b.shape)

# Optionally: print a small portion of A and b
A_dense_small = A_csr.toarray()
print("\nPorción top-left de A (5x5):\n", A_dense_small[:5,:5])
print("\nPorción de b (primeros 10):\n", b[:10])

# Guardar A y b a archivos si querés
# sparse.save_npz("A_ensamblada.npz", A_csr)
# np.savetxt("b_ensamblada.csv", b, delimiter=",")
