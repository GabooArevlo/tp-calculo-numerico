# FEM_Punto6.py
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# ---------------------------
#  (A) NODOS (usa los nodos definidos en Punto 1)
# ---------------------------
nodes = [
    (0.0, 0.0),
    (0.25, 0.0),
    (0.5, 0.0),
    (0.75, 0.0),
    (1.0, 0.0),
    (1.0, 0.25),
    (1.0, 0.5),
    (1.0, 0.75),
    (1.0, 1.0),
    (0.75, 1.0),
    (0.5, 1.0),
    (0.25, 1.0),
    (0.0, 1.0),
    (0.0, 0.75),
    (0.0, 0.5),
    (0.0, 0.25),
    (0.25, 0.25),
    (0.75, 0.25),
    (0.25, 0.75),
    (0.75, 0.75),
    #  Neumann
    (0.5 + 0.3333/2.0, 0.5),
    (0.5 - 0.3333/2.0, 0.5),
    (0.5, 0.5 + 0.3333/2.0),
    (0.5, 0.5 - 0.3333/2.0),
    # Externo
    (0.5, 0.5),
    # interior
    (0.5, 0.25),
    (0.5, 0.75),
    (0.25, 0.5)
]
coords = np.array(nodes)
n_nodes = coords.shape[0]

# ---------------------------
#  (B) Clasificación simple: Dirichlet , Externo, Neumann 
# ---------------------------
center = np.array([0.5, 0.5])
r = 0.3333 / 2.0
tol = 1e-5

is_dirichlet = np.isclose(coords[:,0], 0.0) | np.isclose(coords[:,0], 1.0) | \
               np.isclose(coords[:,1], 0.0) | np.isclose(coords[:,1], 1.0)
dist = np.linalg.norm(coords - center, axis=1)
is_externo = dist < r - tol
is_neumann = np.isclose(dist, r, atol=1e-4)
# uso nodos validos
valid_mask = ~is_externo
coords_valid = coords[valid_mask]
global_index_of_valid = np.where(valid_mask)[0]

# triangulación Delaunay en los nodos válidos
tri = Delaunay(coords_valid)
elements_local = tri.simplices.copy()   # indices locales en coords_valid
elements = []
for tri_local in elements_local:
    elements.append(tuple(global_index_of_valid[tri_local]))  # indices globales

# ---------------------------
#  (C) funciones útiles: Ke y fe (element stiffness y element load)
# ---------------------------
def element_stiffness_matrix(x1, y1, x2, y2, x3, y3):
    A = 0.5 * np.linalg.det(np.array([
        [1, x1, y1],
        [1, x2, y2],
        [1, x3, y3]
    ]))
    if A <= 0:
        raise ValueError("Área no positiva en elemento")
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1
    b = np.array([b1, b2, b3])
    c = np.array([c1, c2, c3])
    Ke = (1.0 / (4.0 * A)) * (np.outer(b, b) + np.outer(c, c))
    return Ke, A

def element_load_vector(x1, y1, x2, y2, x3, y3, f_func):
    # regla simple: fe_i = A * f(xc,yc) * (1/3)  (ya que \int phi_i = A/3)
    xc = (x1 + x2 + x3) / 3.0
    yc = (y1 + y2 + y3) / 3.0
    A = 0.5 * np.linalg.det(np.array([
        [1, x1, y1],
        [1, x2, y2],
        [1, x3, y3]
    ]))
    fe = np.ones(3) * (A * f_func(xc, yc) / 3.0)
    return fe

# ---------------------------
# (D) función f(x,y) (forzante) y g en Neumann (si distinto de cero)
# ---------------------------
def fa(x, y):
    # Ejemplo
    # uso fa consistente con ua=sin(pi x) sin(pi y)
    return 2 * (np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)

def g_neumann_on_edge(xm, ym):
    # Ejemplo si g != 0; si el problema tiene g=0 devolvé 0
    return 0.0

# ---------------------------
# (E) Ensamblado global A y b
# ---------------------------
N_unknowns = np.sum(~is_externo)  # nodos que participan
# mapping global node index -> eq index 
idx_map = -np.ones(n_nodes, dtype=int)
counter = 0
for k in range(n_nodes):
    if not is_externo[k]:
        idx_map[k] = counter
        counter += 1

A = np.zeros((N_unknowns, N_unknowns))
b = np.zeros(N_unknowns)

# Loop sobre elementos
for el in elements:
    # el es tupla de 3 índices globales
    n0, n1, n2 = el
    x1, y1 = coords[n0]
    x2, y2 = coords[n1]
    x3, y3 = coords[n2]

    Ke, Area = element_stiffness_matrix(x1, y1, x2, y2, x3, y3)
    fe = element_load_vector(x1, y1, x2, y2, x3, y3, fa)

    # ensamblar en A y b
    local_nodes = [n0, n1, n2]
    # indices globales en sistema reducido
    for i_local, ni in enumerate(local_nodes):
        Ii = idx_map[ni]
        if Ii == -1:
            continue
        b[Ii] += fe[i_local]
        for j_local, nj in enumerate(local_nodes):
            Jj = idx_map[nj]
            if Jj == -1:
                continue
            A[Ii, Jj] += Ke[i_local, j_local]

# ---------------------------
# (F) Aplicar condiciones de Neumann sobre aristas del agujero (si g != 0)
# ---------------------------
# Encontrar aristas (pares de nodos) que approximan la circunferencia: aristas que tengan ambos nodos is_neumann True
edges = set()
for el in elements:
    tri_nodes = list(el)
    pairs = [(tri_nodes[0], tri_nodes[1]), (tri_nodes[1], tri_nodes[2]), (tri_nodes[2], tri_nodes[0])]
    for a,bn in pairs:
        if a < bn:
            edge = (a,bn)
        else:
            edge = (bn,a)
        edges.add(edge)

for edge in edges:
    a, bnode = edge
    if is_neumann[a] and is_neumann[bnode]:
        # longitud de la arista
        xa, ya = coords[a]
        xb, yb = coords[bnode]
        Ledge = np.hypot(xa - xb, ya - yb)
        # integrar g * phi 
        # uso midpoint rule simpler: 
        xm, ym = (xa + xb)/2.0, (ya + yb)/2.0
        gmid = g_neumann_on_edge(xm, ym)
        # contributions to b 
        Ia = idx_map[a]; Ib = idx_map[bnode]
        if Ia != -1:
            b[Ia] += (Ledge/2.0) * gmid * 0.5  # distribuir
        if Ib != -1:
            b[Ib] += (Ledge/2.0) * gmid * 0.5

# ---------------------------
# (G) Imponer condiciones Dirichlet: sustituir filas/columnas (valores = 0 en frontera exterior)
# ---------------------------
u = np.zeros(N_unknowns)
dirichlet_nodes = np.where(is_dirichlet)[0]
for nd in dirichlet_nodes:
    idx = idx_map[nd]
    if idx == -1:
        continue
    # imposoner u=0 : poner diagonal 1 y b=0
    A[idx, :] = 0.0
    A[idx, idx] = 1.0
    b[idx] = 0.0

# ---------------------------
# (H) Resolver el sistema
# ---------------------------
U = np.linalg.solve(A, b)

# reconstruir solución por nodos (poner nan para Externo)
u_full = np.full(n_nodes, np.nan)
for k in range(n_nodes):
    if idx_map[k] != -1:
        u_full[k] = U[idx_map[k]]

print("Solución nodal (u) por nodo global index:\n", u_full)

# ---------------------------
# (I) Graficar solución (scatter + triangulación)
# ---------------------------
plt.figure(figsize=(7,6))
# plot triangulation 
for el in elements:
    pts = coords[list(el)+[el[0]]]
    plt.plot(pts[:,0], pts[:,1], 'k-', linewidth=0.8)

sc = plt.scatter(coords[:,0], coords[:,1], c=u_full, cmap='viridis', s=60, edgecolors='k')
plt.colorbar(sc, label='u_h nodal')
plt.title('Solución FEM P1 (discretización y valores nodales)')
plt.gca().set_aspect('equal')
plt.show()
