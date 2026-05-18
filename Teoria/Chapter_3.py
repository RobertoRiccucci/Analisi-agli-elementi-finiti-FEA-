import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
import pygmsh

# ------- Consideriamo un dominio a forma di L, con vertici A,B,C,D,E,F, e troviamo Phf, posto f = sin(πx)sin(πy) ---------

# Dominio

A = np.array([0.0, 0.0])
B = np.array([2.0, 0.0])
C = np.array([2.0, 1.0])
D = np.array([1.0, 1.0])
E = np.array([1.0, 2.0])
F = np.array([0.0, 2.0])

# Facciamo il mesh del dominio

with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ],
        mesh_size=0.1,
    )
    mesh = geom.generate_mesh()

P = mesh.points[:, :2] # matrice dei punti 
T = mesh.cells_dict["triangle"] # matrice di connettività

print(f"Matrice dei punti{P}") # Nella teoria si considera la trasposta di questa matrice, ma per il codice torna più utile così
print(f"Matrice dei punti{T}") # Nella teoria si considera la trasposta di questa matrice, ma per il codice torna più utile così

# Funzione presa in esame 

def f(x,y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# Array dell'area dei triangoli 

Trasp_T = mesh.cells_dict["triangle"].T
n_t = np.size(Trasp_T[1]) # numero triangoli

K = np.zeros(n_t)
for i in range(n_t):
    r, s, t = T[i]
    K_i = 0.5 * abs(P[r][0]*(P[s][1] - P[t][1]) + P[s][0]*(P[t][1] - P[r][1]) + P[t][0]*(P[r][1] - P[s][1])) # area del triangolo K_i
    K[i] =  K_i

# Troviamo la matrice M

Trasp_P = mesh.points[:, :2].T
n_p = np.size(Trasp_P[1])

M_loc_template = np.array([[2, 1, 1],
                           [1, 2, 1],
                           [1, 1, 2]]) # matrice di massa locale (template, va moltiplicata per |K| / 12)

M_K_glob = np.zeros((n_t, n_p, n_p))

for i in range(n_t):
    r, s, t = T[i]
    nodi = [r, s, t]
    
    M_loc = M_loc_template * K[i] / 12
    
    for j in range(3):
        for k in range(3):
            M_K_glob[i, nodi[j], nodi[k]] = M_loc[j, k]

M = M_K_glob.sum(axis=0) # matrice globale come somma di tutti i contributi

# Troviamo il vettore b

b_K_glob = np.zeros((n_t, n_p))

for i in range(n_t):
    nodi = T[i]
    
    for j in range(3):
        N_j = P[nodi[j]]
        b_K_glob[i, nodi[j]] = f(N_j[0], N_j[1]) * K[i] / 3

b = b_K_glob.sum(axis=0) # vettore globale come somma di tutti i contributi

# Risolviamo il sistema 
Phf = np.linalg.solve(M, b)

# -------- Plot dei risultati ----------

triang = mtri.Triangulation(P[:, 0], P[:, 1], T)

# Valori di f esatta ai nodi (per confronto)
f_nodi = f(P[:, 0], P[:, 1])

# Scala di colori comune
vmin = min(f_nodi.min(), Phf.min())
vmax = max(f_nodi.max(), Phf.max())
livelli = np.linspace(vmin, vmax, 21)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# f esatta
cs1 = axes[0].tricontourf(triang, f_nodi, levels=livelli, cmap="viridis")
axes[0].triplot(triang, lw=0.2, color="k", alpha=0.3)
axes[0].set_title(r"$f(x,y) = \sin(\pi x)\sin(\pi y)$")
axes[0].set_aspect("equal")

# Proiezione P_h f
cs2 = axes[1].tricontourf(triang, Phf, levels=livelli, cmap="viridis")
axes[1].triplot(triang, lw=0.2, color="k", alpha=0.3)
axes[1].set_title(r"$P_h f$")
axes[1].set_aspect("equal")

# Colorbar unica condivisa
fig.colorbar(cs2, ax=axes, shrink=0.8, label="valore")

plt.show()