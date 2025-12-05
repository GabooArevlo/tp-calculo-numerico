# ==========================================
# PLANTEO ANALÍTICO - VERSIÓN CORREGIDA
# ==========================================
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1️⃣ Definir variables simbólicas
x, y = sp.symbols('x y')

# 2️⃣ Definir funciones u_d y u_n según el enunciado
u_d = x * (1 - x) * y * (1 - y)
# El radio del círculo es 0.3333/2 ≈ 1/6
r_squared = (1/6)**2 
u_n = ((x - 0.5)**2 + (y - 0.5)**2 - r_squared)**2

# 3️⃣ Definir la función analítica combinada (CORRECCIÓN: Se usa PRODUCTO)
u_a = u_d * u_n
u_a_simpl = sp.simplify(u_a)
print("Función analítica ua(x,y) = ")
sp.pprint(u_a_simpl)

# ============ Punto 1: Verificar condiciones de borde ============
print("\n--- Verificación de condiciones de borde ---")

## Condición 1: Dirichlet (u_a = 0) en Γd (Contorno exterior)
# Puntos representativos: (0, 0.5), (1, 0.5), (0.5, 0), (0.5, 1)
borde_exterior = [
    (0, 0.5),    # x=0
    (1, 0.5),    # x=1
    (0.5, 0),    # y=0
    (0.5, 1)     # y=1
]

print("Verificación de Dirichlet (u_a=0) en Γd:")
for (xx, yy) in borde_exterior:
    val = u_a.subs({x: xx, y: yy})
    # Debe ser 0.0 (debido al factor u_d)
    print(f"u_a({xx},{yy}) =", float(val)) 

## Condición 2: Neumann (∂u_a/∂n = 0) en Γn (Círculo interior)
r = 0.3333 / 2 
r_sym = sp.Rational(1, 6) # Usamos 1/6 para precisión simbólica

# Componentes del gradiente de u_a
ux = sp.diff(u_a, x)
uy = sp.diff(u_a, y)

# Definición paramétrica del círculo (normales n = (cosθ, sinθ))
θ = sp.symbols('θ')
x_circ = 0.5 + r_sym * sp.cos(θ)
y_circ = 0.5 + r_sym * sp.sin(θ)

# Derivada normal (∂u/∂n) = ∇u · n = ux*cosθ + uy*sinθ
# NOTA: SimPy simplificará esto a cero debido a que u_n=0 y ∂u_n/∂n=0 en el borde.
un = ux.subs({x:x_circ, y:y_circ}) * sp.cos(θ) + uy.subs({x:x_circ, y:y_circ}) * sp.sin(θ)
un_simpl = sp.simplify(un)
print("\nVerificación de Neumann (∂u_a/∂n=0) en Γn:")
print("Derivada normal en el círculo (∂u/∂n):")
sp.pprint(un_simpl) 
# Resultado esperado: 0

# ============ Punto 2: Calcular f_a(x,y) = Δu_a ============
# f_a(x,y) es el Laplaciano de u_a
lap_u_a = sp.diff(u_a, x, 2) + sp.diff(u_a, y, 2)
lap_u_a_simpl = sp.simplify(lap_u_a)

print("\nFunción fuente f_a(x,y) = Δu_a =")
sp.pprint(lap_u_a_simpl)

# ============ Punto 3: Graficar u_a y f_a en 3D ============
u_a_func = sp.lambdify((x, y), u_a_simpl, "numpy")
f_a_func = sp.lambdify((x, y), lap_u_a_simpl, "numpy")

# Generar la malla
X = np.linspace(0, 1, 100)
Y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(X, Y)

# Calcular los valores de Z, enmascarando el círculo interior si es necesario para f_a
Z_u = u_a_func(X, Y)
Z_f = f_a_func(X, Y)

# Crear la figura (como la que mostraste)
fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(14, 6))

# Gráfico 1: Solución analítica u_a(x,y)
axs[0].plot_surface(X, Y, Z_u, cmap='viridis')
axs[0].set_title("Solución analítica $u_a(x,y)$")
axs[0].set_xlabel("x"); axs[0].set_ylabel("y"); axs[0].set_zlabel("$u_a$")

# Gráfico 2: Función fuente f_a(x,y)
axs[1].plot_surface(X, Y, Z_f, cmap='plasma')
axs[1].set_title("Función fuente $f_a(x,y) = \Delta u_a$")
axs[1].set_xlabel("x"); axs[1].set_ylabel("y"); axs[1].set_zlabel("$f_a$")

plt.tight_layout()
plt.show()