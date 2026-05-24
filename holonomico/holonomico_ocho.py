import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parámetros
a = 5
dt = 0.001

# posición inicial del agente
pos = np.array([1.0,0.0])  # shape (1,2)
vel = np.zeros_like(pos)

def F_8(x, y, a=a):
    return x**2 - 4*y**2*(1 - y**2)

def grad_F_8(x, y, a=None):
    dFdx = 2*x
    dFdy = 16*y**3 - 8*y
    return np.array([dFdx, dFdy])

# definición del gvf generalizado
def gvf(pos_i, F, grad_F, k = 1):
    x, y = pos_i

    phi = F(x, y)
    grad = grad_F(x, y)

    # normal
    vector_normal = -k * phi * grad

    # tangente (rotamos 90% al ser 2d)
    vector_tangente = np.array([-grad[1], grad[0]])

    # vector resultado
    V = vector_normal + vector_tangente

    # normalizamos el resulºtado para que no afecte su magnitud
    norm = np.linalg.norm(V)
    if norm != 0:
        V = V / norm
        
    return V

# malla más densa
X, Y = np.meshgrid(
    np.linspace(-2, 2, 40),
    np.linspace(-2, 2, 40)
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
#Esta mal parametrizada
#Lo cambio
'''
w = np.linspace(0, 2*np.pi, 400)
x_8 = 2 * a * np.cos(w) * np.sin(w)
y_8 = a * np.sin(w)
'''
w = np.linspace(0, 2*np.pi, 400)
x_8 = np.sin(2*w)  # Esta es la forma natural de x^2 = 4y^2(1-y^2)
y_8 = np.sin(w)

fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(X, Y, U, V, color='black')
ax.plot(x_8, y_8, 'r', label='Trayectoria')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)

# scatter del agente
scatter = ax.scatter(pos[0], pos[1], c='b', s=50, label='Agente')
ax.legend()

# función de actualización de agente
def update_agent(frame):
    speed=25;
    global pos
    # calcular GVF
    V = gvf(pos, F_8, grad_F_8) 
    # actualizar posición
    pos += V * dt* speed
    scatter.set_offsets(pos)
    return scatter,

# animación
ani = FuncAnimation(fig, update_agent, frames=400, interval=50, blit=True)
plt.show()
