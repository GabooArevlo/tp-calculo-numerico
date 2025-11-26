import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. MALLA REGULAR NxN
# ============================================================
N = 20
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
hx = 1/(N-1)
hy = 1/(N-1)

X, Y = np.meshgrid(x, y)
num_nodes = N*N

def idx(i, j):
    return i*N + j

# ============================================================
# 2. SOLUCIÓN ANALÍTICA
# ============================================================
def u_analitica(x, y):
    return x*y*(x-1)*(y-1) - ((x-0.5)**4 + (y-0.5)**4 - 0.0277777777777778)

def f_fuente(x, y):
    return -14*x**2 + 14*x - 14*y**2 + 14*y - 7.77777777777778

U_exacta = u_analitica(X, Y)

# ============================================================
# 3. MATRIZ GLOBAL (Laplaciano 5 puntos)
# ============================================================
A = np.zeros((num_nodes, num_nodes))
b = np.zeros(num_nodes)

for i in range(N):
    for j in range(N):

        k = idx(i, j)
        xk = X[i, j]
        yk = Y[i, j]

        # Términos fuente
        b[k] = f_fuente(xk, yk)

        # Caso borde → se impondrá después
        if i == 0 or i == N-1 or j == 0 or j == N-1:
            continue

        # FEM simplificado (≈ DF 5 puntos)
        A[k, idx(i, j)] = -2/(hx*hx) - 2/(hy*hy)
        A[k, idx(i+1, j)] = 1/(hx*hx)
        A[k, idx(i-1, j)] = 1/(hx*hx)
        A[k, idx(i, j+1)] = 1/(hy*hy)
        A[k, idx(i, j-1)] = 1/(hy*hy)

# ============================================================
# 4. CONDICIONES DE BORDE DIRICHLET
# ============================================================
for i in range(N):
    for j in range(N):

        if i == 0 or i == N-1 or j == 0 or j == N-1:

            k = idx(i, j)
            A[k, :] = 0
            A[k, k] = 1
            b[k] = u_analitica(X[i, j], Y[i, j])

# ============================================================
# 5. RESOLVER SISTEMA LINEAL (Punto 9)
# ============================================================
U_h = np.linalg.solve(A, b)
U_h = U_h.reshape((N, N))

# ============================================================
# 6. GRÁFICO 3D DE LA SOLUCIÓN NUMÉRICA
# ============================================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U_h, linewidth=0, antialiased=True)
ax.set_title("Solución Numérica FEM (Punto 9)")
plt.show()

# ============================================================
# 7. ERROR (Punto 9–10)
# ============================================================
Error = np.abs(U_h - U_exacta)
error_max = np.max(Error)
error_promedio = np.mean(Error)

print("\n========= ERRORES FEM =========")
print("Error absoluto máximo =", error_max)
print("Error absoluto promedio =", error_promedio)

# ============================================================
# 8. GRÁFICO 3D DEL ERROR
# ============================================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Error, linewidth=0, antialiased=True)
ax.set_title("Error absoluto |u_h - u_a| (Punto 10)")
plt.show()
