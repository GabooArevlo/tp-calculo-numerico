import numpy as np

def element_stiffness_matrix(x1, y1, x2, y2, x3, y3):
    """
    Devuelve la matriz de rigidez (K_e) de un elemento triangular
    con nodos en coordenadas arbitrarias.
    """

    # --- 1) Área del triángulo ---
    A = 0.5 * np.linalg.det(np.array([
        [1, x1, y1],
        [1, x2, y2],
        [1, x3, y3]
    ]))

    if A <= 0:
        raise ValueError("El triángulo tiene área cero o está orientado al revés.")

    # --- 2) Coeficientes geométricos ---
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2

    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    b = np.array([b1, b2, b3])
    c = np.array([c1, c2, c3])

    # --- 3) Matriz elemental ---
    Ke = (1 / (4 * A)) * (np.outer(b, b) + np.outer(c, c))

    return Ke, A


# ========================
# EJEMPLO DE USO
# ========================
if __name__ == "__main__":
    # Triángulo arbitrario (como los de tu imagen)
    x1, y1 = 0.0, 0.0   # nodo a
    x2, y2 = 2.0, 0.5   # nodo b
    x3, y3 = 0.5, 1.5   # nodo c

    Ke, A = element_stiffness_matrix(x1, y1, x2, y2, x3, y3)

    print("Área del triángulo:", A)
    print("\nMatriz de rigidez K_e:\n", Ke)
