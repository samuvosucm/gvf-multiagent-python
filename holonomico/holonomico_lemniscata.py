import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parámetros
a = 5
dt = 0.02

# posición inicial del agente
pos = np.array([2.0, 0.0])  # shape (1,2)
vel = np.zeros_like(pos)

def F_lemniscata(x, y, a=a):
    return (x**2 + y**2)**2 - 2*a**2*(x**2 - y**2)

def grad_F_lemniscata(x, y, a=a):
    dFdx = 4*x*(x**2 + y**2 - a**2)
    dFdy = 4*y*(x**2 + y**2 + a**2)
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

# dibujar vectores del gvf
X, Y = np.meshgrid(np.linspace(-8, 8, 20), np.linspace(-8, 8, 20))
U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos_point = np.array([X[i,j], Y[i,j]])
        vec = gvf(pos_point, F_lemniscata, grad_F_lemniscata)
        U[i,j] = vec[0]
        V[i,j] = vec[1]


# dibujamos la lemniscata como trayectoria
w = np.linspace(0, 2*np.pi, 400)
x_lem = a * np.cos(w) / (1 + np.sin(w)**2)
y_lem = a * np.sin(w) * np.cos(w) / (1 + np.sin(w)**2)

fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(X, Y, U, V, color='blue')
ax.plot(x_lem, y_lem, 'r', label='Trayectoria')
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
    V = gvf(pos, F_lemniscata, grad_F_lemniscata) 
    # actualizar posición
    pos += V * dt
    scatter.set_offsets(pos)
    return scatter,

# animación
ani = FuncAnimation(fig, update_agent, frames=400, interval=50, blit=True)
plt.show()
