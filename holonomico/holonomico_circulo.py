import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cvxpy as cp

# parámetros
r = 5
dt = 0.1

# posición inicial del agente
pos = np.array([
    [0.0, 0.0]
])
vel = np.zeros_like(pos)

# definición del gvf
def gvf(pos_i, r, k = 0.1):
    x, y = pos_i

    # tangente (rotando 90 grados)
    vector_tangente = np.array([-2*y, 2*x])

    # normal
    phi = x**2 + y**2 - r**2
    vector_normal = -k * phi * np.array([2*x, 2*y])

    # vector resultadod4
    V = vector_normal + vector_tangente

    # normalizamos el resultado para que no afecte su magnitud
    norm = np.linalg.norm(V)

    if norm < 1e-8:
        return np.array([0.0, 0.0])

    V = V / norm
    return V

# función de seguridad CBF
def h(xi, xj, R):
    return np.linalg.norm(xi - xj)**2 - R**2

def cbf_qp(u_des, pos_i, pos, i, R=1.5, alpha=1.0):
    u = cp.Variable(2)

    constraints = []

    for j in range(len(pos)):
        if i == j:
            continue

        diff = pos_i - pos[j]
        h_ij = np.dot(diff, diff) - R**2

        grad_h = 2 * diff

        # CBF constraint:
        constraints.append(grad_h @ u + alpha * h_ij >= 0)

    # objetivo: estar lo más cerca posible de u_des
    objective = cp.Minimize(cp.sum_squares(u - u_des))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if u.value is None:
        return u_des  # fallback si falla

    return u.value

    return correction

# malla más densa
X, Y = np.meshgrid(
    np.linspace(-8, 8, 40),
    np.linspace(-8, 8, 40)
)


# dibujamos el círculo como trayectoria
w = np.linspace(0, 2*np.pi, 400)
x_circle = r*np.cos(w)
y_circle = r*np.sin(w)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_circle, y_circle, 'r', label='Trayectoria')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid(True)

# scatter del agente
scatter = ax.scatter(pos[:, 0], pos[:, 1], c='b', s=50, label='Agente')
ax.legend()

# función de actualización de agente
def update_agent(frame):
    global pos

    new_pos = pos.copy()

    for i in range(len(pos)):

        u_des = gvf(pos[i], r)

        u_safe = cbf_qp(u_des, pos[i], pos, i)

        new_pos[i] += u_safe * dt

    pos = new_pos
    scatter.set_offsets(pos)

    return scatter,

# animación
ani = FuncAnimation(fig, update_agent, frames=400, interval=50, blit=True)
plt.show()
