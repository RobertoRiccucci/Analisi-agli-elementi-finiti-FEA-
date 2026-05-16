import numpy as np
from matplotlib import pyplot as plt

# ------ Andiamo a trovare Phf, posto f = sin(x) considerando n nodi equidistanti a due a due -------- 

# Consideriamo l'intervallo I = [0,1]
x_a, x_b = [0, 10]
n = 100
lunghezza_l = x_b - x_a
H = np.linspace(x_a, x_b, n+1) # array dei nodi
h = lunghezza_l/n # lunghezza di ciascun sottointervallo 

# Funzione presa in esame
def f(x):
    return np.sin(x)

# Troviamo la matrice M 
M = np.zeros((n+1,n+1))

for i in range(n):
    M[i][i] += h/3
    M[i+1][i] += h/6
    M[i][i+1] += h/6
    M[i+1][i+1] += h/3

# Troviamo il vettore b
b = np.zeros(n+1)

for i in range(n):
    b[i] += f(H[i])*h/2
    b[i+1] += f(H[i+1])*h/2

# Risolviamo il sistema 

Phf = np.linalg.solve(M, b)

print(f"il vettore Phf scritto in base \phi_i è: {Phf}")

#Plot dei risultati 
asse_x = np.linspace(x_a, x_b, 1000)

plt.plot(asse_x, f(asse_x), label = "$f(x) = \\sin(x)$")
plt.errorbar(H, Phf, fmt = ".",label = "$Phf_{coeff}$")
plt.legend()
plt.show()