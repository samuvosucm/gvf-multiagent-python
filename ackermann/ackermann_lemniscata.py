import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import matplotlib.transforms as transforms
import cvxpy as cp

# parámetros
a = 8.0
dt = 0.1
L = 2.5

states = np.array([
    [2.0, 0.0, 0.0, 0.0], # x, y, w, theta
    [-2.0, 1.5, 0.0, 0.0],


])


v_nominal = np.array([1.1, 2])
R_safe = 1.5

def trajectory(w):
    x = a * np.cos(w) / (1 + np.sin(w)**2)
    y = a * np.sin(w) * np.cos(w) / (1 + np.sin(w)**2)
    return np.array([x, y])

def dtrajectory_dw(w):
    eps = 1e-5
    return (trajectory(w + eps) - trajectory(w)) / eps

def gvf_augmented(x, y, w, k=1.0):

    xd, yd = trajectory(w)
    dx_dw, dy_dw = dtrajectory_dw(w)

    ex = x - xd
    ey = y - yd

    # normal correction
    V_norm = -k * np.array([ex, ey])

    # tangent direction in physical space
    tangent = np.array([dx_dw, dy_dw])
    tangent = tangent / (np.linalg.norm(tangent) + 1e-6)

    # combine
    V_xy = tangent + V_norm
    V_xy = tangent + V_norm
    V_xy = np.clip(V_xy, -5, 5)
    vw = np.dot([ex, ey], tangent)

    return np.array([V_xy[0], V_xy[1], vw])

def cbf_ackermann(i, states, v_des, phi_des, L, R=1.5, alpha=2.0):
    x, y, w, theta = states[i]

    v = cp.Variable()
    phi = cp.Variable()

    constraints = []

    for j in range(len(states)):
        if i == j:
            continue

        xj, yj, _, _ = states[j]

        # función de barrera
        diff = np.array([x, y]) - np.array([xj, yj])
        h = np.dot(diff, diff) - R**2
        
        grad_h = 2 * diff

        vx = v * np.cos(theta)
        vy = v * np.sin(theta)

        constraints.append(
            grad_h[0] * vx + grad_h[1] * vy + alpha * h >= 0
        )

    # límites físicos
    constraints += [
        v >= 0,
        v <= 2.0,
        phi >= -np.pi/4,
        phi <= np.pi/4
    ]

    objective = cp.Minimize((v - v_des)**2 + (phi - phi_des)**2)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if v.value is None:
        return v_des, phi_des

    return v.value, phi.value


t = np.linspace(0, 2*np.pi, 500)
x = a * np.cos(t) / (1 + np.sin(t)**2)
y = a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, 'r')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid(True)

cars = []
for _ in range(len(states)):
    car = Polygon(np.array([
        [0.5, 0],
        [-0.5, -0.3],
        [-0.5, 0.3]
    ]), closed=True, fc='b')
    ax.add_patch(car)
    cars.append(car)

def update(frame):
    global states

    new_states = states.copy()

    for i in range(len(states)):

        x, y, w, theta = states[i]

        # GVF aumentado
        V = gvf_augmented(x, y, w)

        vx, vy, vw = V

        # dirección deseada
        theta_d = np.arctan2(vy, vx)

        error = np.arctan2(np.sin(theta_d - theta),
                           np.cos(theta_d - theta))

        v_des = v_nominal[i]
        
        omega = 2.0 * error

        phi_des = np.arctan((L * omega) / (v_des + 1e-6))
        phi_des = np.clip(phi_des, -np.pi/4, np.pi/4)

        v_safe, phi_safe = cbf_ackermann(
            i, states, v_des, phi_des, L
        )

        # dinámica
        x += v_safe * np.cos(theta) * dt
        y += v_safe * np.sin(theta) * dt
        theta += (v_safe / L) * np.tan(phi_safe) * dt

        # actualizar parámetro interno
        w += vw * dt

        new_states[i] = [x, y, w, theta]

        # dibujo
        tform = (transforms.Affine2D()
                 .rotate_around(0, 0, theta)
                 .translate(x, y)
                 + ax.transData)

        cars[i].set_transform(tform)

    states[:] = new_states

    return cars

ani = FuncAnimation(fig, update, frames=400, interval=50, blit=False)
plt.show()