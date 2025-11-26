import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ================================================================
#  FUNCIÓN ANALÍTICA Y FUENTE
# ================================================================
def u_analitica(x, y):
    return x*y*(x-1)*(y-1) - ((x-0.5)**4 + (y-0.5)**4 - 0.0277777777777778)

def f_fuente(x, y):
    return -14*x**2 + 14*x - 14*y**2 + 14*y - 7.77777777777778


# ================================================================
#  GENERADOR DE MALLA GMSH
# ================================================================
def generar_malla(NN, filename):

    gmsh.initialize()
    gmsh.model.add("dominio")

    # El dominio es el cuadrado unitario
    lc = 1 / np.sqrt(NN)  

    gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)

    gmsh.finalize()



# ================================================================
#  ENSAMBLADO FEM LINEAL P1
# ================================================================
def resolver_fem(filename):

    m = meshio.read(filename)

    # nodos
    pts = m.points[:, :2]
    num_nodes = len(pts)

    # conectividades triangulares
    tri = m.cells_dict["triangle"]

    A = np.zeros((num_nodes, num_nodes))
    b = np.zeros(num_nodes)

    for T in tri:
        p1, p2, p3 = T
        x1, y1 = pts[p1]; x2, y2 = pts[p2]; x3, y3 = pts[p3]

        # matriz jacobiana
        J = np.array([[x2 - x1, x3 - x1],
                      [y2 - y1, y3 - y1]])
        detJ = np.linalg.det(J)
        area = detJ / 2

        # gradientes locales de las funciones base
        grad = np.array([[-1, -1],
                         [ 1,  0],
                         [ 0,  1]]) @ np.linalg.inv(J)

        # matriz local
        Ke = area * (grad @ grad.T)

        # ensamblado global
        for a in range(3):
            for c in range(3):
                A[T[a], T[c]] += Ke[a, c]

        # vector fuente (promedio)
        xm = (x1 + x2 + x3)/3
        ym = (y1 + y2 + y3)/3
        fe = f_fuente(xm, ym) * area / 3

        for a in range(3):
            b[T[a]] += fe


    # Condiciones de borde Dirichlet en nodos del contorno
    tol = 1e-12
    for i, (x, y) in enumerate(pts):
        if x < tol or x > 1-tol or y < tol or y > 1-tol:
            A[i, :] = 0
            A[i, i] = 1
            b[i] = u_analitica(x, y)

    # resolver
    Uh = np.linalg.solve(A, b)
    return pts, tri, Uh


# ================================================================
#  GRAFICAR UNA SUPERFICIE DE TRIANGULOS
# ================================================================
def grafico_3d(pts, tri, values, title):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    X = pts[:,0]
    Y = pts[:,1]
    Z = values

    for T in tri:
        ax.plot_trisurf(X[T], Y[T], Z[T], linewidth=0.2, antialiased=True)

    ax.set_title(title)
    plt.show()


# ================================================================
#  BUCLE PRINCIPAL (PUNTOS 11–12–13)
# ================================================================
mallados = [49, 81, 121, 441, 961, 1681, 2601]

error_max_list = []
error_prom_list = []

for NN in mallados:

    print(f"\n=== MALLA DE {NN} NODOS ===")

    filename = f"malla_{NN}.msh"
    generar_malla(NN, filename)

    pts, tri, Uh = resolver_fem(filename)

    # solución exacta
    Uex = u_analitica(pts[:,0], pts[:,1])

    # errores
    error = np.abs(Uh - Uex)
    error_max = np.max(error)
    error_prom = np.mean(error)

    error_max_list.append(error_max)
    error_prom_list.append(error_prom)

    print("Error máximo   =", error_max)
    print("Error promedio =", error_prom)

    # graficar solución FEM
    grafico_3d(pts, tri, Uh, f"Solución FEM - {NN} nodos")

    # graficar error
    grafico_3d(pts, tri, error, f"Error absoluto - {NN} nodos")


# ================================================================
#  GRÁFICOS DE CONVERGENCIA (Punto 13)
# ================================================================
plt.figure(figsize=(7,5))
plt.plot(mallados, error_max_list, marker='o')
plt.xlabel("Cantidad de nodos")
plt.ylabel("Error absoluto máximo")
plt.title("Convergencia FEM (error máximo)")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
plt.plot(mallados, error_prom_list, marker='o')
plt.xlabel("Cantidad de nodos")
plt.ylabel("Error absoluto promedio")
plt.title("Convergencia FEM (error promedio)")
plt.grid(True)
plt.show()
