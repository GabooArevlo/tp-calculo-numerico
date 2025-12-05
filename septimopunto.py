import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Necesario para el gráfico 3D

# --- 1. CONFIGURACIÓN DEL MALLADO (Reutilización de datos) ---
# Usamos las coordenadas y la matriz U_grid_2D obtenidas en el punto 6
# N_total = 49, n = 7, coords, U_grid_2D
n = 7
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)
coords = np.column_stack((X.flatten(), Y.flatten()))

# Copia la matriz de solución U_grid_2D obtenida en el paso 6:
U_grid_2D = np.array([
    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.00028599, -0.00016034, -0.00030845, -0.00016034,  0.00028599,  0.        ],
    [ 0.        , -0.00016034, -0.00056174, -0.00059162, -0.00056174, -0.00016034,  0.        ],
    [ 0.        , -0.00030845, -0.00059162,          np.nan, -0.00059162, -0.00030845,  0.        ],
    [ 0.        , -0.00016034, -0.00056174, -0.00059162, -0.00056174, -0.00016034,  0.        ],
    [ 0.        ,  0.00028599, -0.00016034, -0.00030845, -0.00016034,  0.00028599,  0.        ],
    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ]
])

# --- 2. EVALUACIÓN DE LA SOLUCIÓN ANALÍTICA u_a(x,y) ---
x_sym, y_sym = sp.symbols('x y')
r_squared_sym = sp.Rational(1, 6)**2

# Solución analítica u_a = u_d * u_n
u_d = x_sym * (1 - x_sym) * y_sym * (1 - y_sym)
u_n = ((x_sym - 0.5)**2 + (y_sym - 0.5)**2 - r_squared_sym)**2
u_a = u_d * u_n

# Función lambdify para evaluar u_a
u_a_func = sp.lambdify((x_sym, y_sym), u_a, "numpy")

# Evaluar u_a en todos los nodos de la malla
U_analytic_2D = u_a_func(X, Y)

# --- 3. CÁLCULO DEL ERROR ---
# Error absoluto E_A = | u_a - u_aprox |
# Se ignora el nodo nan del centro (externo)
E_abs_2D = np.abs(U_analytic_2D - U_grid_2D)

# Eliminar NaNs para calcular promedios y máximos solo sobre los nodos del dominio
valid_mask = ~np.isnan(E_abs_2D)
E_abs_valid = E_abs_2D[valid_mask]

# Cálculo de Métricas (Requeridas para el Punto 10)
E_MAX = np.max(E_abs_valid)
E_promedio = np.mean(E_abs_valid)

print("\n===== MÉTRICAS DE ERROR (Malla 7x7) =====")
print(f"Error Absoluto Máximo (E_MAX): {E_MAX:.6e}")
print(f"Error Absoluto Promedio: {E_promedio:.6e}")
print("=========================================")

# --- 4. GRÁFICO 3D DEL ERROR ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Usar la malla X, Y para las coordenadas y E_abs_2D para la altura.
# Los NaNs se ignoran automáticamente en el plot 3D de NumPy.
ax.plot_surface(X, Y, E_abs_2D, cmap='viridis', edgecolor='none')

ax.set_title(f'Error Absoluto |u_a - u_aprox| (Malla 7x7)\nE_MAX: {E_MAX:.2e}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Error Absoluto')
plt.show()