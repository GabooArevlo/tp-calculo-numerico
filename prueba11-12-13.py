# ============================================================
# CÓDIGO FINAL FEM (Corregido)
# ============================================================

import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse.linalg as spla
import sympy as sp
import os

# ----------------------------------------------------------------
#  1. DEFINICIÓN ANALÍTICA Y FUENTE (CORREGIDA)
# ----------------------------------------------------------------
x_sym, y_sym = sp.symbols('x y')
r_hole_num = 0.3333 / 2.0
r_hole_sym = sp.Rational(3333, 20000)

# u_d cumple Dirichlet (0 en contorno exterior)
u_d_sym = x_sym * (1 - x_sym) * y_sym * (1 - y_sym)
# u_n cumple Neumann (0 en contorno interior)
u_n_sym = ((x_sym - 0.5)**2 + (y_sym - 0.5)**2 - r_hole_sym**2)**2

# CORRECCIÓN 1: u_a = u_d * u_n para cumplir ambas CC y ser no negativa.
u_a_sym = u_d_sym * u_n_sym

# -----------------------------------------------------------
# CÁLCULO DE LA FUNCIÓN FUENTE f_FEM = -Δu_a
laplaciano_u_a = sp.diff(u_a_sym, x_sym, 2) + sp.diff(u_a_sym, y_sym, 2)
# CORRECCIÓN 2: f_fuente = -Laplaciano(u_a) para resolver -Δu = f
f_fuente_sym = sp.simplify(-laplaciano_u_a) 
# -----------------------------------------------------------

# Funciones numéricas (lambdify)
u_analitica = sp.lambdify((x_sym, y_sym), u_a_sym, "numpy")
f_fuente = sp.lambdify((x_sym, y_sym), f_fuente_sym, "numpy")

# Gradiente para Neumann
u_x_num = sp.lambdify((x_sym, y_sym), sp.diff(u_a_sym, x_sym), "numpy")
u_y_num = sp.lambdify((x_sym, y_sym), sp.diff(u_a_sym, y_sym), "numpy")

center = np.array([0.5, 0.5])

# C.B. Neumann gN = Grad(u_a) . n
def gN_at_point(xp, yp):
    dx = xp - center[0]; dy = yp - center[1]
    distancia = np.hypot(dx, dy)
    # Evitar división por cero si el punto es exactamente el centro (aunque no debería ocurrir en el borde)
    if np.isclose(distancia, 0.0, atol=1e-8): return 0.0 
    # Vector normal unitario n
    nx = dx / distancia; ny = dy / distancia
    # Derivada Normal = Grad(u) . n
    return u_x_num(xp, yp) * nx + u_y_num(xp, yp) * ny

# ----------------------------------------------------------------
#  2. GENERADOR DE MALLA GMSH 
# ----------------------------------------------------------------
def generar_malla_anillo(NN_target, filename):
    
    tol = 1e-6 
    
    if not gmsh.isInitialized():
        gmsh.initialize()
    else:
        gmsh.clear() 
    
    gmsh.model.add("anillo")
    gmsh.option.setNumber("General.Verbosity", 1)
    
    # lc (Tamaño característico de la malla) basado en el número de nodos
    lc = 1.0 / (np.sqrt(NN_target) / 3.0) 
    
    # Cuadrado exterior y Círculo interior
    square = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    circle = gmsh.model.occ.addDisk(0.5, 0.5, 0.0, r_hole_num, r_hole_num)
    
    # Operación de Sustracción (Anillo)
    gmsh.model.occ.cut([(2, square)], [(2, circle)])
    gmsh.model.occ.synchronize()

    # Identificación del Dominio (Superficie 2D)
    surfaces = gmsh.model.getEntities(2)
    if not surfaces:
        raise ValueError("GMSH no pudo crear la superficie del anillo.")
    s1 = surfaces[0][1] 
    
    # Obtener todas las curvas de frontera (1D)
    boundary_entities = gmsh.model.getBoundary([(2, s1)], combined=False)
    
    dirichlet_lines = []
    neumann_circle = []
    
    # Clasificación de curvas por su centroide
    for dim, tag in boundary_entities:
        if dim == 1: 
            params = gmsh.model.getParametrizationBounds(dim, tag)
            p_mid = (params[0] + params[1]) / 2.0
            
            xyz_mid = gmsh.model.getValue(dim, tag, p_mid)
            mid_point = np.array([xyz_mid[0], xyz_mid[1]])
            
            distance_to_center = np.linalg.norm(mid_point - center)
            
            # Neumann (interior)
            if np.isclose(distance_to_center, r_hole_num, atol=0.01):
                neumann_circle.append(tag)
            # Dirichlet (exterior)
            elif np.isclose(mid_point[0], 0.0, atol=tol) or np.isclose(mid_point[0], 1.0, atol=tol) or \
                 np.isclose(mid_point[1], 0.0, atol=tol) or np.isclose(mid_point[1], 1.0, atol=tol):
                dirichlet_lines.append(tag)
    
    if not neumann_circle or not dirichlet_lines:
        raise ValueError("GMSH falló en identificar las fronteras Dirichlet/Neumann.")

    # Asignar Physical Groups
    gmsh.model.addPhysicalGroup(1, dirichlet_lines, tag=101) # Dirichlet (Exterior)
    gmsh.model.addPhysicalGroup(1, neumann_circle, tag=102) # Neumann (Interior)
    gmsh.model.addPhysicalGroup(2, [s1], tag=201) # Dominio

    # Establecer el tamaño de la malla en las entidades
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    # Generación y Finalización
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()


# ================================================================
#  ENSAMBLADO FEM LINEAL P1 
# ================================================================

def resolver_fem(filename):
    
    tol = 1e-6
    
    try:
        m = meshio.read(filename)
    except Exception as e:
        raise ValueError(f"Error al leer la malla con meshio: {e}")

    pts = m.points[:, :2]
    num_nodes = len(pts)
    tri = m.cells_dict.get("triangle", np.array([]))
    if not tri.any():
        raise ValueError("No se encontraron elementos triangulares en la malla.")

    from scipy.sparse import lil_matrix
    A = lil_matrix((num_nodes, num_nodes))
    b = np.zeros(num_nodes)
    
    # 1. Ensamblaje de Rigidez (A) y Carga de Volumen (b)
    for T in tri:
        p1, p2, p3 = T
        x1, y1 = pts[p1]; x2, y2 = pts[p2]; x3, y3 = pts[p3]

        J = np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
        detJ = np.linalg.det(J)
        area = detJ / 2

        if area <= tol: continue 

        # Gradientes de las funciones base locales
        grad = np.array([[-1, -1], [ 1, 0], [ 0, 1]]) @ np.linalg.inv(J)
        # Matriz de Rigidez elemental: Ke = ∫ grad(phi_i) . grad(phi_j) dΩ ≈ Area * grad.T @ grad
        Ke = area * (grad @ grad.T)

        # Vector de Carga elemental (Método de cuadratura de 1 punto en el centroide)
        xm = (x1 + x2 + x3)/3; ym = (y1 + y2 + y3)/3
        fe = f_fuente(xm, ym) * area / 3 # f_fuente se usa como f_FEM en el sistema Ku=F

        for a in range(3):
            b[T[a]] += fe
            for c in range(3):
                A[T[a], T[c]] += Ke[a, c]

    # 2. Condición de Borde de NEUMANN (Aporte al vector b)
    neumann_edges_indices = []
    
    if "gmsh:physical" in m.cell_data_dict and 'line' in m.cells_dict:
        line_data = m.cell_data_dict["gmsh:physical"]["line"]
        line_cells = m.cells_dict["line"]
        
        for i, tag in enumerate(line_data):
            if tag == 102: # Tag del Círculo Interior
                neumann_edges_indices.append((line_cells[i][0], line_cells[i][1]))
    
    for n1, n2 in neumann_edges_indices:
        xa, ya = pts[n1]; xb, yb = pts[n2]
        Ledge = np.hypot(xa - xb, ya - yb)
        xm, ym = 0.5*(xa+xb), 0.5*(ya+yb)
        
        gmid = gN_at_point(xm, ym) # gN(x,y) = Grad(u_a).n
        # Integral de Neumann: ∫ gN * phi_i dΓ ≈ Ledge * gN_mid * phi_i_mid
        contrib = Ledge * gmid 
        
        b[n1] += contrib * 0.5 # phi_1(mid) = 0.5
        b[n2] += contrib * 0.5 # phi_2(mid) = 0.5

    # 3. Condición de Borde de DIRICHLET (Imposición fuerte)
    dirichlet_nodes = []
    for i, (x, y) in enumerate(pts):
        # Nodos en el contorno exterior del cuadrado [0, 1]x[0, 1]
        if np.isclose(x, 0.0, atol=tol) or np.isclose(x, 1.0, atol=tol) or \
           np.isclose(y, 0.0, atol=tol) or np.isclose(y, 1.0, atol=tol):
            dirichlet_nodes.append(i)

    # Imponer u = u_analitica en el contorno exterior
    for i in dirichlet_nodes:
        # Fila de A a cero, A[i,i] = 1, b[i] = u_analitica
        A[i, :] = 0
        A[i, i] = 1
        b[i] = u_analitica(pts[i, 0], pts[i, 1])

    # 4. Resolver el sistema lineal Ku = F
    Uh = spla.spsolve(A.tocsr(), b)
    return pts, tri, Uh


# ================================================================
#  GRAFICAR UNA SUPERFICIE DE TRIANGULOS
# ================================================================

def grafico_3d(pts, tri, values, title):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot_trisurf(pts[:,0], pts[:,1], values, triangles=tri, cmap='viridis', linewidth=0.2, antialiased=True)

    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('Valor')
    plt.show()


# ================================================================
#  BUCLE PRINCIPAL (PUNTOS 11–12–13)
# ================================================================

mallados_target = [49, 81, 121, 441, 961, 1681, 2601]

error_max_list = []; error_prom_list = []
nodes_count_list = [] 

# Inicializar GMSH
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 1)

for NN_target in mallados_target:
    print(f"\n=== MALLA OBJETIVO {NN_target} NODOS ===")

    filename = f"malla_{NN_target}_anillo.msh"
    
    try:
        generar_malla_anillo(NN_target, filename)
        
        pts, tri, Uh = resolver_fem(filename)
        
    except ValueError as e:
        print(f"ERROR: Falló la generación/resolución de la malla {NN_target}: {e}")
        continue
    except Exception as e:
        print(f"ERROR: Falló inesperado para la malla {NN_target}: {e}")
        continue

    # Post-proceso
    N_real = len(pts)
    nodes_count_list.append(N_real)

    Uex = u_analitica(pts[:,0], pts[:,1])
    error = np.abs(Uh - Uex)
    error_max = np.max(error)
    error_prom = np.mean(error)

    error_max_list.append(error_max)
    error_prom_list.append(error_prom)

    print(f"Nodos reales = {N_real}")
    print(f"Error máximo   = {error_max:.8f}")
    print(f"Error promedio = {error_prom:.8f}")

    # Gráficos (Punto 12)
    grafico_3d(pts, tri, Uh, f"Solución FEM - {N_real} nodos")
    # El gráfico de error absoluto siempre será positivo
    grafico_3d(pts, tri, error, f"Error absoluto - {N_real} nodos") 
    
    if os.path.exists(filename):
        os.remove(filename)

# ----------------------------------------------------------------
#  6. GRÁFICOS DE CONVERGENCIA (Punto 13)
# ----------------------------------------------------------------

if gmsh.isInitialized():
    gmsh.finalize()
    
if nodes_count_list:
    # Error Máximo
    plt.figure(figsize=(7,5))
    plt.plot(nodes_count_list, error_max_list, marker='o') 
    plt.xlabel("Cantidad de nodos")
    plt.ylabel("Error absoluto máximo")
    plt.title("Convergencia FEM (error máximo)")
    plt.grid(True)
    plt.show()

    # Error Promedio
    plt.figure(figsize=(7,5))
    plt.plot(nodes_count_list, error_prom_list, marker='o')
    plt.xlabel("Cantidad de nodos")
    plt.ylabel("Error absoluto promedio")
    plt.title("Convergencia FEM (error promedio)")
    plt.grid(True)
    plt.show()
else:
    print("\nNo se pudieron generar datos de convergencia debido a errores de ejecución.")