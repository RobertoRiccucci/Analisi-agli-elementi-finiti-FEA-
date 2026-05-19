import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
import pygmsh

# ------- Consideriamo un dominio Ω, con vertici A,B,C,D e troviamo una soluzione dell'equazione di Poisson ---------

# Dominio

A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([1.0, 1.0])
D = np.array([0.0, 1.0])

# Facciamo il mesh del dominio

with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        mesh_size=0.1,
    )
    mesh = geom.generate_mesh()

P = mesh.points[:, :2] # matrice dei punti 
T = mesh.cells_dict["triangle"] # matrice di connettività

# Funzione presa in esame
def f(x,y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Soluzione esatta (per il confronto)
def u_exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# Array dell'area dei triangoli 

Trasp_T = mesh.cells_dict["triangle"].T
n_t = np.size(Trasp_T[1]) # numero triangoli

K = np.zeros(n_t)
for i in range(n_t):
    r, s, t = T[i]
    K_i = 0.5 * abs(P[r][0]*(P[s][1] - P[t][1]) + P[s][0]*(P[t][1] - P[r][1]) + P[t][0]*(P[r][1] - P[s][1])) # area del triangolo K_i
    K[i] =  K_i

# Troviamo la matrice A 

Trasp_P = mesh.points[:, :2].T
n_p = np.size(Trasp_P[1])

A_K_glob = np.zeros((n_t, n_p, n_p))

for i in range(n_t):
    r, s, t = T[i]
    nodi = [r, s, t]
    
    # Coefficienti delle hat functions locali
    b_coef = np.array([P[s][1] - P[t][1],
                       P[t][1] - P[r][1],
                       P[r][1] - P[s][1]]) / (2 * K[i])
    c_coef = np.array([P[t][0] - P[s][0],
                       P[r][0] - P[t][0],
                       P[s][0] - P[r][0]]) / (2 * K[i])
    
    A_loc = (np.outer(b_coef, b_coef) + np.outer(c_coef, c_coef)) * K[i]

    for j in range(3):
        for k in range(3):
            A_K_glob[i, nodi[j], nodi[k]] = A_loc[j, k]

A = A_K_glob.sum(axis=0) # matrice globale come somma di tutti i contributi

# Troviamo il vettore b

b_K_glob = np.zeros((n_t, n_p))

for i in range(n_t):
    nodi = T[i]
    
    for j in range(3):
        N_j = P[nodi[j]]
        b_K_glob[i, nodi[j]] = f(N_j[0], N_j[1]) * K[i] / 3

b = b_K_glob.sum(axis=0) # vettore globale come somma di tutti i contributi

# Identificazione dei nodi di bordo
spigoli_bordo = mesh.cells_dict["line"] 
I_b = np.unique(spigoli_bordo) # indici dei nodi di bordo
I_int = np.setdiff1d(np.arange(n_p), I_b) # nodi interni

# Sottoblocchi della matrice di rigidezza
A_00 = A[np.ix_(I_int, I_int)] # blocco interno-interno
A_0g = A[np.ix_(I_int, I_b)] # blocco interno-bord 

# Sottovettore del termine noto
b_0  = b[I_int]

# Rilolviamo il sistema (non consideriamo la parte con A_0g perché siamo nel caso omogeneo)

u_int = np.linalg.solve(A_00, b_0)
u_h = np.zeros(n_p)
u_h[I_int] = u_int

# -------- Plot dei risultati --------

# Soluzione esatta valutata ai nodi
u_ex = u_exact(P[:, 0], P[:, 1])

# Triangolazione per matplotlib
triang = mtri.Triangulation(P[:, 0], P[:, 1], T)

# Scala di colori condivisa per u_ex e u_h
vmin = min(u_ex.min(), u_h.min())
vmax = max(u_ex.max(), u_h.max())
livelli = np.linspace(vmin, vmax, 21)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# u esatta
cs1 = axes[0].tricontourf(triang, u_ex, levels=livelli, cmap="viridis")
axes[0].triplot(triang, lw=0.2, color="k", alpha=0.3)
axes[0].set_title(r"$u(x,y) = \sin(\pi x)\sin(\pi y)$ (esatta)")
axes[0].set_aspect("equal")
fig.colorbar(cs1, ax=axes[0], shrink=0.8)

# u_h FEM
cs2 = axes[1].tricontourf(triang, u_h, levels=livelli, cmap="viridis")
axes[1].triplot(triang, lw=0.2, color="k", alpha=0.3)
axes[1].set_title(r"$u_h$ (FEM)")
axes[1].set_aspect("equal")
fig.colorbar(cs2, ax=axes[1], shrink=0.8)

plt.tight_layout()
plt.show()