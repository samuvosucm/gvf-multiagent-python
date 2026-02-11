import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parámetros
a = 5
dt = 0.1

# posición inicial del agente
xi = np.array([2.0, 0.0, 0.0])  # x, y, w
vel = np.zeros_like(xi)

def f(w):
    x = 2 * a * np.cos(w) * np.sin(w)
    y = a * np.sin(w)
    return np.array([x, y])

def phi(xi):
    x, y, w = xi
    fx, fy = f(w)
    return np.array([x - fx, y - fy])

# derivadas de f con respecto a w
def df_dw(w):
    dx = 2 * a * np.cos(2*w)
    dy = a * np.cos(w)
    return np.array([dx, dy])

def grad_phi(xi):
    x, y, w = xi
    dfx, dfy = df_dw(w)
    grad1 = np.array([1, 0, -dfx])
    grad2 = np.array([0, 1, -dfy])
    return grad1, grad2

def tangent_vector(grad1, grad2):
    return np.cross(grad1, grad2)

def normal_vector(xi, k=1.0):
    ph = phi(xi)
    grad1, grad2 = grad_phi(xi)
    return -k * (ph[0] * grad1 + ph[1] * grad2)

# definición del gvf generalizado
def gvf_augmented(xi, k=1.0):
    grad1, grad2 = grad_phi(xi)
    V_tan = tangent_vector(grad1, grad2)
    V_norm = normal_vector(xi, k)
    V = V_tan + V_norm
    return V / np.linalg.norm(V)

# dibujamos el 8 como trayectoria
w = np.linspace(0, 2*np.pi, 400)
x_8 = 2 * a * np.cos(w) * np.sin(w)
y_8 = a * np.sin(w)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_8, y_8, 'r', label='Trayectoria')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid(True)

# scatter del agente
scatter = ax.scatter(xi[0], xi[1], c='b', s=50, label='Agente')
ax.legend()

# función de actualización de agente
def update_agent(frame):
    global xi
    V = gvf_augmented(xi)
    xi += V * dt
    scatter.set_offsets([xi[0], xi[1]])
    return scatter,

# animación
ani = FuncAnimation(fig, update_agent, frames=400, interval=50, blit=True)
plt.show()
