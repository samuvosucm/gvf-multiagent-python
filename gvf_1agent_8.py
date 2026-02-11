import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parámetros
a = 5
dt = 0.02

# posición inicial del agente
pos = np.array([2.0, 0.0])  # shape (1,2)
vel = np.zeros_like(pos)

def F_8(x, y, a=a):
    return x**2 - 4*y**2*(1 - y**2)

def grad_F_8(x, y, a=None):
    dFdx = 2*x
    dFdy = 8*y**3 - 8*y
    return np.array([dFdx, dFdy])

# definición del gvf generalizado
def gvf(pos_i, F, grad_F, k = 0.4):
    x, y = pos_i

    phi = F(x, y)
    grad = grad_F(x, y)

    # normal
    vector_normal = -k * phi * grad

    # tangente (rotamos 90% al ser 2d)
    vector_tangente = np.array([-grad[1], grad[0]])

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
            F_8,
            grad_F_8
        )


# dibujamos el 8 como trayectoria
w = np.linspace(0, 2*np.pi, 400)
x_8 = 2 * a * np.cos(w) * np.sin(w)
y_8 = a * np.sin(w)

fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(X, Y, U, V, color='black')
ax.plot(x_8, y_8, 'r', label='Trayectoria')
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
    V = gvf(pos, F_8, grad_F_8) 
    # actualizar posición
    pos += V * dt
    scatter.set_offsets(pos)
    return scatter,

# animación
ani = FuncAnimation(fig, update_agent, frames=400, interval=50, blit=True)
plt.show()
