import numpy as np

def elemento_K(x1, y1, x2, y2, x3, y3):

    A = 0.5 * np.linalg.det(np.array([
        [1, x1, y1],
        [1, x2, y2],
        [1, x3, y3]
    ]))

    if A <= 0:
        raise ValueError("El área del triángulo no es positiva, revisar nodos.")

    
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2

    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    K = (1/(4*A)) * np.array([
        [b1*b1 + c1*c1,   b1*b2 + c1*c2,   b1*b3 + c1*c3],
        [b2*b1 + c2*c1,   b2*b2 + c2*c2,   b2*b3 + c2*c3],
        [b3*b1 + c3*c1,   b3*b2 + c3*c2,   b3*b3 + c3*c3]
    ])

    return K, A

x1, y1 = 0, 0
x2, y2 = 1, 0
x3, y3 = 0, 1

K, A = elemento_K(x1, y1, x2, y2, x3, y3)

print("Área del triángulo =", A)
print("\nMatriz de rigidez del elemento (K):\n")
print(K)
