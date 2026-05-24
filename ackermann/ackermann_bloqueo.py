import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle
import matplotlib.transforms as transforms
import cvxpy as cp

a      = 6
L      = 2.0
v_ref  = 3.0
k      = 1.0
k_hdg  = 2.0
dt     = 0.05

# CBF params
R_safe  = 1.5
alpha   = 2.0
v_max   = 3.0
phi_max = np.pi/3

priority = [1.0, 0.2]

# Estado por coche: [x, y, theta, w]
xis = [
    np.array([ 0.0,  1.0, np.pi/4, 0.0]),
    np.array([-4.0, 1.0, 0.0,     np.pi])
]

colors = ['tab:blue', 'tab:green']

def f(w):
    x = a * np.cos(w) / (1 + np.sin(w)**2)
    y = a * np.sin(w) * np.cos(w) / (1 + np.sin(w)**2)
    return np.array([x, y])

def df_dw(w):
    denom = (1 + np.sin(w)**2)**2

    dx = -a * np.sin(w) * (3 - np.sin(w)**2) / denom

    dy = a * (
        (np.cos(w)**2 - np.sin(w)**2) * (1 + np.sin(w)**2)
        - 2 * np.sin(w)**2 * np.cos(w)**2
    ) / denom

    return np.array([dx, dy])

def phi(s):
    fx, fy = f(s[2])
    return np.array([s[0] - fx, s[1] - fy])

def grad_phi(s):
    dfx, dfy = df_dw(s[2])
    return (np.array([1.0, 0.0, -dfx]),
            np.array([0.0, 1.0, -dfy]))

def gvf(s, k=2.0):
    g1, g2 = grad_phi(s)
    ph     = phi(s)
    V_tan  = np.cross(g1, g2)
    V_norm = -k * (ph[0]*g1 + ph[1]*g2)
    V = V_tan + V_norm
    norm = np.linalg.norm(V)
    return V / norm if norm > 1e-8 else V_tan / np.linalg.norm(V_tan)

def cbf_ackermann_multi(i, states, v_des, phi_des,
                        R=1.5, alpha=2.0,
                        v_max=3.0, phi_max=np.pi/3):

    x, y, theta, w = states[i]

    v = cp.Variable()
    phi = cp.Variable()
    delta = cp.Variable(nonneg=True)

    constraints = []

    for j in range(len(states)):
        if j == i:
            continue

        xj, yj, _, _ = states[j]

        diff = np.array([x - xj, y - yj], dtype=float)
        h = float(diff @ diff - R**2)
        grad_h = 2.0 * diff

        vx = v * np.cos(theta)
        vy = v * np.sin(theta)

        constraints.append(
            grad_h[0]*vx + grad_h[1]*vy + alpha*h >= -delta * priority[i]
        )

    constraints += [
        v >= 0.3,          
        v <= v_max,
        phi >= -phi_max,
        phi <= phi_max
    ]

    objective = cp.Minimize(
        cp.square(v - v_des) +
        0.2 * cp.square(phi - phi_des) +
        300 * delta
    )

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if v.value is None or phi.value is None:
        return float(v_des), float(phi_des)

    return float(v.value), float(phi.value)

def ackermann_step_i(i, states):
    x, y, theta, w = states[i]
    s = np.array([x, y, w])

    V = gvf(s, k)
    vx_gvf, vy_gvf, _ = V

    xy_mag = np.hypot(vx_gvf, vy_gvf)
    if xy_mag > 0.05:
        theta_d = np.arctan2(vy_gvf, vx_gvf)
    else:
        theta_d = theta

    e_theta = np.arctan2(np.sin(theta_d - theta),
                         np.cos(theta_d - theta))

    phi_des = np.clip(np.arctan(L * k_hdg * e_theta),
                       -phi_max, phi_max)

    v_des = v_ref

    v_safe, phi_safe = cbf_ackermann_multi(
        i=i,
        states=states,
        v_des=float(v_des),
        phi_des=float(phi_des),
        R=R_safe,
        alpha=alpha,
        v_max=v_max,
        phi_max=phi_max
    )

    df = df_dw(w)
    df_sq = max(np.dot(df, df), 1e-8)

    v_xy = v_safe * np.array([np.cos(theta), np.sin(theta)])
    w_dot = np.dot(v_xy, df) / df_sq

    return np.array([
        v_safe * np.cos(theta),
        v_safe * np.sin(theta),
        v_safe * np.tan(phi_safe) / L,
        w_dot
    ])

w_traj = np.linspace(0, 2*np.pi, 800)
x_t, y_t = f(w_traj)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_t, y_t, 'r', lw=2, label='Trayectoria')

ax.set_xlim(-8, 8)
ax.set_ylim(-7, 7)
ax.set_aspect('equal')
ax.grid(True)

safe_circles = []
for c in colors:
    circ = Circle((0, 0), R_safe, fill=False, ec=c, ls='--', lw=1, alpha=0.8)
    ax.add_patch(circ)
    safe_circles.append(circ)

cars = []
base_shape = np.array([
    [0.5, 0.0],
    [-0.5, -0.3],
    [-0.5, 0.3]
])

for c in colors:
    poly = Polygon(base_shape.copy(), closed=True, fc=c, ec='k', alpha=0.9)
    ax.add_patch(poly)
    cars.append(poly)

def set_car_transform(patch, x, y, theta):
    tform = (transforms.Affine2D()
             .rotate_around(0, 0, theta)
             .translate(x, y)
             + ax.transData)
    patch.set_transform(tform)

def update(frame):
    global xis

    states_now = [xi.copy() for xi in xis]

    steps = []
    for i in range(len(xis)):
        steps.append(ackermann_step_i(i, states_now))

    for i in range(len(xis)):
        xis[i] = xis[i] + steps[i] * dt
        xis[i][2] = np.arctan2(np.sin(xis[i][2]),
                               np.cos(xis[i][2]))

    artists = []
    for i, xi in enumerate(xis):
        x, y, theta, w = xi
        set_car_transform(cars[i], x, y, theta)
        safe_circles[i].center = (x, y)
        artists += [cars[i], safe_circles[i]]

    return artists

ani = FuncAnimation(fig, update, frames=1200, interval=30, blit=True)
plt.title("GVF + CBF (solucionado)")
plt.show()