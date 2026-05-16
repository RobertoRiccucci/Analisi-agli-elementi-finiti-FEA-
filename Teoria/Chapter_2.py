import numpy as np
from matplotlib import pyplot as plt

# Consideriamo l'intervallo I = [0, 1]
x_a, x_b = 0, 1
n = 100
lunghezza_I = x_b - x_a
H = np.linspace(x_a, x_b, n+1) # array dei nodi (n+1 nodi)
h = lunghezza_I / n  # lunghezza di ciascun sottointervallo

# Funzione presa in esame
def f(x):
    return np.pi**2 * np.sin(np.pi * x)

# Soluzione esatta (per il confronto)
def u_exact(x):
    return np.sin(np.pi * x)

# Troviamo la matrice A 
A = np.zeros((n+1, n+1))
for i in range(n):
    A[i,   i]   += 1/h
    A[i+1, i]   -= 1/h
    A[i,   i+1] -= 1/h
    A[i+1, i+1] += 1/h

# Troviamo il vettore b 
b = np.zeros(n+1)
for i in range(n):
    b[i]   += f(H[i])   * h / 2
    b[i+1] += f(H[i+1]) * h / 2

# ---- Imposizione delle condizioni di Dirichlet u(0) = u(1) = 0 ----
# Togliamo la prima e l'ultima riga/colonna del sistema
A_int = A[1:-1, 1:-1]
b_int = b[1:-1]

# Risoluzione del sistema ridotto
u_int = np.linalg.solve(A_int, b_int)

#Ricostruzione della soluzione completa, includendo i nodi di bordo
u_h = np.zeros(n+1)
u_h[1:-1] = u_int

# Plot dei risultati
asse_x = np.linspace(x_a, x_b, 1000)

plt.plot(asse_x, u_exact(asse_x), label=r"$u(x) = \sin(\pi x)$ (esatta)")
plt.errorbar(H, u_h, fmt=".", label=r"$u_h$ (FEM)")
plt.legend()
plt.show()