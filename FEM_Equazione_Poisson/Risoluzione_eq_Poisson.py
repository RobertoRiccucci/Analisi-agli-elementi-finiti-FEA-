import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
import pygmsh

# ------- Consideriamo un dominio Ω e troviamo una soluzione dell'equazione di Poisson poste le condizioni al contorno di Dirichlet ---------

np.random.seed(42)  # seed per riproducibilità dei risultati

# Dominio casuale semplicemente connesso (fatto sulla stile random walk)
n_modi = 5
R0 = 1.0
a_modi = np.random.uniform(-0.2, 0.2, n_modi)
phi_modi = np.random.uniform(0, 2*np.pi, n_modi)

theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
raggio = R0 + sum(a_modi[k] * np.cos((k+1)*theta + phi_modi[k]) for k in range(n_modi))
vertici = np.column_stack([raggio*np.cos(theta), raggio*np.sin(theta)])

# Condizioni al bordo casuali 
cA, cB, cC, cD, cE = np.random.uniform(-1, 1, 5)

def g_D(x, y):
    return cA*x + cB*y + cC*np.cos(np.pi*x) + cD*np.sin(np.pi*y) + cE*np.exp(x*y) # formula a caso

# Densità di carica (doppio dipolo)
def rho(x, y):
    sigma = 0.05
    x1, y1 = -0.3, 0.0   # carica positiva
    x2, y2 =  0.3, 0.0   # carica negativa
    Q = 1.0
    g1 = np.exp(-((x-x1)**2 + (y-y1)**2) / (2*sigma**2))
    g2 = np.exp(-((x-x2)**2 + (y-y2)**2) / (2*sigma**2))
    eps_0 = 8.85 * 10**(-12)
    return Q / (2*np.pi*sigma**2 * eps_0) * (g1 - g2)

# Generazione della mesh
with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(vertici.tolist(), mesh_size=0.05)
    mesh = geom.generate_mesh()

P = mesh.points[:, :2] # matrice dei punti 
T = mesh.cells_dict["triangle"] # matrice di connettività
Trasp_T = mesh.cells_dict["triangle"].T
Trasp_P = mesh.points[:, :2].T
n_p = np.size(Trasp_P[1]) # numero nodi
n_t = np.size(Trasp_T[1]) # numero triangoli

# Array dell'area dei triangoli 

K = np.zeros(n_t)
for i in range(n_t):
    r, s, t = T[i]
    K_i = 0.5 * abs(P[r][0]*(P[s][1] - P[t][1]) + P[s][0]*(P[t][1] - P[r][1]) + P[t][0]*(P[r][1] - P[s][1])) # area del triangolo K_i
    K[i] =  K_i

# Troviamo la matrice A 

A_stiff = np.zeros((n_p, n_p))

for i in range(n_t):
    r, s, t = T[i]
    nodi = [r, s, t]

    # Coefficienti delle hat functions locali
    b_coef = np.array([P[s,1] - P[t,1],
                       P[t,1] - P[r,1],
                       P[r,1] - P[s,1]]) / (2 * K[i])
    c_coef = np.array([P[t,0] - P[s,0],
                       P[r,0] - P[t,0],
                       P[s,0] - P[r,0]]) / (2 * K[i])

    # Matrice di rigidezza locale \tilde{A}^K
    A_loc = (np.outer(b_coef, b_coef) + np.outer(c_coef, c_coef)) * K[i]

    # Assemblaggio: accumulo direttamente sulle entrate globali
    for j in range(3):
        for k in range(3):
            A_stiff[nodi[j], nodi[k]] += A_loc[j, k]

# Troviamo il vettore b
b_vec = np.zeros(n_p)

for i in range(n_t):
    nodi = T[i]
    for j in range(3):
        N_j = P[nodi[j]]
        b_vec[nodi[j]] += rho(N_j[0], N_j[1]) * K[i] / 3

# Identificazione dei nodi di bordo
spigoli_bordo = mesh.cells_dict["line"]
I_b   = np.unique(spigoli_bordo)
I_int = np.setdiff1d(np.arange(n_p), I_b)

# Partizione del sistema
A_00 = A_stiff[np.ix_(I_int, I_int)] # blocco interno-interno
A_0g = A_stiff[np.ix_(I_int, I_b)] # blocco interno-bordo
b_0  = b_vec[I_int]

# Valori al bordo (g_D valutata sui nodi di bordo della mesh)
xi_g = g_D(P[I_b, 0], P[I_b, 1])

# Risoluzione del sistema ridotto
u_int = np.linalg.solve(A_00, b_0 - np.dot(A_0g, xi_g))

# Ricostruzione della soluzione globale
u_h = np.zeros(n_p)
u_h[I_int] = u_int
u_h[I_b] = xi_g

# ------- Plot dei risultati -------
triang = mtri.Triangulation(P[:, 0], P[:, 1], T)

fig, ax = plt.subplots(figsize=(8, 7))
cs = ax.tricontourf(triang, u_h, levels=30, cmap="RdBu_r")
ax.triplot(triang, lw=0.2, color="k", alpha=0.3)
ax.set_aspect("equal")
ax.set_title(r"Potenziale $u_h$ su dominio casuale, dipolo + Dirichlet non omogenee")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(cs, ax=ax, label=r"$u_h$ (unità arbitrarie)")
plt.tight_layout()
plt.show()