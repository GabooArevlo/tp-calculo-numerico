# =========================================================
# PUNTOS 1 y 2 EF: Definición Manual de Nodos y Mallado
# =========================================================
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# --- Definición de Geometría ---
center = (0.5, 0.5)
r = 0.3333 / 2.0
tol = 1e-6 # Aumentar la tolerancia para chequeos

# --- 1. Definición Manual de Coordenadas (N=28 total, 27 válidos) ---
# Nodos: 16 Dirichlet + 7 Internos + 4 Neumann + 1 Externo
nodes = [
    # Frontera Exterior (Dirichlet)
    (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
    (1.0, 0.25), (1.0, 0.5), (1.0, 0.75), (1.0, 1.0),
    (0.75, 1.0), (0.5, 1.0), (0.25, 1.0), (0.0, 1.0),
    (0.0, 0.75), (0.0, 0.5), (0.0, 0.25),
    # Nodos Internos (Interior del dominio)
    (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
    (0.5, 0.25), (0.5, 0.75), (0.25, 0.5),
    # Nodos Neumann (Sobre el círculo interior r=0.16665)
    (0.5 + r, 0.5), (0.5 - r, 0.5), (0.5, 0.5 + r), (0.5, 0.5 - r),
    # Nodo Externo (Centro del Hueco)
    (0.5, 0.5), # Este nodo será excluido del mallado
]

coords = np.array(nodes)
n_nodes = coords.shape[0]

# --- 2. Clasificación de Nodos (Punto 1) ---
types = []
distances = np.sqrt((coords[:,0]-center[0])**2 + (coords[:,1]-center[1])**2)

for (xk, yk), d in zip(coords, distances):
    # Condición Dirichlet: Perímetro cuadrado exterior
    if np.isclose(xk, 0.0) or np.isclose(xk, 1.0) or np.isclose(yk, 0.0) or np.isclose(yk, 1.0):
        types.append("Dirichlet")
    # Condición Externo: Dentro del hueco
    elif d < r - tol:
        types.append("Externo")
    # Condición Neumann: Sobre la frontera curva interior (r)
    elif np.isclose(d, r, atol=tol):
        types.append("Neumann")
    # Condición Interno: Resto del dominio
    else:
        types.append("Interno")

df_nodes = pd.DataFrame({
    "N° nodo": np.arange(n_nodes),
    "x": coords[:,0],
    "y": coords[:,1],
    "Tipo": types
})

print("==============================")
print("  TABLA DE NODOS (Punto 1)")
print("==============================")
print(df_nodes.to_string())
print("\nConteo por tipo:\n", df_nodes["Tipo"].value_counts())
print(f"Total Nodos: {n_nodes} (Válidos: {n_nodes - df_nodes['Tipo'].value_counts().get('Externo', 0)})")


# --- 3. Mallado de Elementos Finitos (Punto 2) ---

# Excluir nodos "Externo" para la triangulación
valid_mask = df_nodes["Tipo"] != "Externo"
coords_valid = coords[valid_mask]
indices_map = np.where(valid_mask)[0] # Mapeo de índice en coords_valid al índice original

# Triangulación de Delaunay
tri = Delaunay(coords_valid)

elements = []
for t in tri.simplices:
    # Mapear los índices locales (t[i]) a los índices originales (n0, n1, n2)
    n0 = indices_map[t[0]]
    n1 = indices_map[t[1]]
    n2 = indices_map[t[2]]
    elements.append((n0, n1, n2))

df_elem = pd.DataFrame(elements, columns=["Nodo 1", "Nodo 2", "Nodo 3"])
df_elem.index.name = "N° elemento"

print("\n==============================")
print("  TABLA DE ELEMENTOS (Punto 2)")
print("==============================")
print(df_elem)
print(f"\nTotal de elementos generados: {len(df_elem)}")

# --- Visualización del Mallado ---
plt.figure(figsize=(7, 7))
plt.triplot(coords_valid[:,0], coords_valid[:,1], tri.simplices, color='gray')
plt.plot(coords_valid[:,0], coords_valid[:,1], 'o', markerfacecolor='blue', markeredgecolor='k', markersize=5)
plt.title(f'Mallado EF (N={len(coords_valid)} nodos válidos)')
plt.xlabel('x'); plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.show()
