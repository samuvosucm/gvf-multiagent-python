import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import matplotlib.transforms as transforms

r = 5
dt = 0.1

state = np.array([2.0, 0.0, 0.0])  # x, y, thetaz
v = 1.0

def gvf(pos_i, r, k=0.02):
    x, y = pos_i
    vector_tangente = np.array([-2*y, 2*x])
    phi = x**2 + y**2 - r**2
    vector_normal = -k * phi * np.array([2*x, 2*y])
    V = vector_tangente + vector_normal
    theta_d = np.arctan2(V[1], V[0])
    return theta_d

w = np.linspace(0, 2*np.pi, 400)
x_circle = r*np.cos(w)
y_circle = r*np.sin(w)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_circle, y_circle, 'r', label='Trayectoria')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid(True)

# dimensiones del coche (triángulo)
car_length = 1.0
car_width = 0.5

triangle_coords = np.array([[car_length/2, 0],    # punta
                            [-car_length/2, -car_width/2],  # base izquierda
                            [-car_length/2, car_width/2]])  # base derecha

car_patch = Polygon(triangle_coords, closed=True, fc='b', ec='black')
ax.add_patch(car_patch)
ax.legend()

def update_agent(frame):
    global state, car_patch

    x, y, theta = state

    # dirección deseada (GVF)
    theta_d = gvf([x, y], r)

    # error angular
    error_theta = np.arctan2(np.sin(theta_d - theta), np.cos(theta_d - theta))

    # velocidad angular para uniciclo
    k_theta = 2.0
    omega = k_theta * error_theta

    # modelo cinemático de uniciclo
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += omega * dt

    state[:] = [x, y, theta]

    # rotar y mover el triángulo
    t = transforms.Affine2D().rotate_around(0, 0, theta) + transforms.Affine2D().translate(x, y) + ax.transData
    car_patch.set_transform(t)

    return [car_patch]
    
ani = FuncAnimation(fig, update_agent, frames=400, interval=50, blit=True)
plt.show()
