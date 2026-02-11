import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parámetros
r = 5
dt = 0.1

# posición inicial del agente
pos = np.array([2.0, 0.0])  # shape (1,2)
vel = np.zeros_like(pos)

# definición del gvf
def gvf(pos_i, r, k = 0.1):
    x, y = pos_i

    # tangente (rotando 90 grados)
    vector_tangente = np.array([-2*y, 2*x])

    # normal
    phi = x**2 + y**2 - r**2
    vector_normal = -k * phi * np.array([2*x, 2*y])

    # vector resultado
    V = vector_normal + vector_tangente

    # normalizamos el resultado para que no afecte su magnitud
    V = V / np.linalg.norm(V)

    return V

# malla más densa
X, Y = np.meshgrid(
    np.linspace(-8, 8, 40),
    np.linspace(-8, 8, 40)
)

# calcular GVF
U, V = np.zeros_like(X), np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        U[i, j], V[i, j] = gvf(
            np.array([X[i, j], Y[i, j]]),
            r
        )

# dibujamos el círculo como trayectoria
w = np.linspace(0, 2*np.pi, 400)
x_circle = r*np.cos(w)
y_circle = r*np.sin(w)

fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(X, Y, U, V, color='black')
ax.plot(x_circle, y_circle, 'r', label='Trayectoria')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid(True)

# scatter del agente
scatter = ax.scatter(pos[0], pos[1], c='b', s=50, label='Agente')
ax.legend()

# función de actualización de agente
def update_agent(frame):
    global pos
    # calcular GVF
    V = gvf(pos, r)   # pos[0] porque solo hay un agente
    # actualizar posición
    pos += V * dt
    scatter.set_offsets(pos)
    return scatter,

# animación
ani = FuncAnimation(fig, update_agent, frames=400, interval=50, blit=True)
plt.show()
