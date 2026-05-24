import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import matplotlib.transforms as transforms

# parámetros
a      = 5
L      = 2.0
v_ref  = 3.0
k      = 1.0
k_hdg  = 2.0
dt     = 0.05

xi = np.array([2.0, 0.0, np.pi / 4, 0.0])

# geometría
def f(w):
    return np.array([a * np.sin(3*w), a * np.sin(2*w)])

def df_dw(w):                    # derivada analítica exacta
    return np.array([3*a * np.cos(3*w),
                     2*a * np.cos(2*w)])

def phi(s):
    fx, fy = f(s[2])
    return np.array([s[0] - fx, s[1] - fy])

def grad_phi(s):
    dfx, dfy = df_dw(s[2])
    return (np.array([1.0, 0.0, -dfx]),
            np.array([0.0, 1.0, -dfy]))

def gvf(s, k=1.0):
    g1, g2 = grad_phi(s)
    ph     = phi(s)
    V_tan  = np.cross(g1, g2)
    V_norm = -k * (ph[0]*g1 + ph[1]*g2)
    V = V_tan + V_norm
    norm = np.linalg.norm(V)
    # Fallback al tangente puro (norma ≥ 1 siempre) si algo sale mal
    return V / norm if norm > 1e-8 else V_tan / np.linalg.norm(V_tan)

def ackermann_step(xi):
    x, y, theta, w = xi
    s = np.array([x, y, w])

    V = gvf(s, k)
    vx, vy, _ = V
    xy_mag = np.hypot(vx, vy)
    if xy_mag > 0.05:               
        theta_d = np.arctan2(vy, vx)
    else:
        theta_d = theta                     

    e_theta = np.arctan2(np.sin(theta_d - theta),
                         np.cos(theta_d - theta))

    phi_s = np.clip(np.arctan(L * k_hdg * e_theta), -np.pi/3, np.pi/3)

    df   = df_dw(w)
    df_sq = max(np.dot(df, df), 1e-8)          # |df/dw|² ≥ ε
    v_xy = v_ref * np.array([np.cos(theta), np.sin(theta)])
    w_dot = np.dot(v_xy, df) / df_sq

    return np.array([
        v_ref * np.cos(theta),
        v_ref * np.sin(theta),
        v_ref * np.tan(phi_s) / L,
        w_dot
    ])

w_traj = np.linspace(0, 2*np.pi, 800)
x_t, y_t = f(w_traj)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_t, y_t, 'r', lw=2, label='Trayectoria')
ax.set_xlim(-8, 8); ax.set_ylim(-7, 7)
ax.set_aspect('equal'); ax.grid(True)

car = Polygon(np.array([
    [0.5, 0],
    [-0.5, -0.3],
    [-0.5, 0.3]
]), closed=True, fc='b')
ax.add_patch(car)
ax.legend()

def update(frame):
    global xi
    xi    += ackermann_step(xi) * dt
    xi[2]  = np.arctan2(np.sin(xi[2]), np.cos(xi[2]))  # wrap θ ∈ (-π, π]

    x, y, theta, _ = xi

    tform = (transforms.Affine2D()
             .rotate_around(0, 0, theta)
             .translate(x, y)
             + ax.transData)

    car.set_transform(tform)

    return car,

ani = FuncAnimation(fig, update, frames=1200, interval=30, blit=True)
plt.title("GVF – Ackermann, Lissajous (3,2)")
plt.show()