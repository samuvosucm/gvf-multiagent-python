import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
a = 5
t = np.linspace(0, 2 * np.pi, 1000)
 
x = (a * np.cos(t)) / (1 + np.sin(t)**2)
y = (a * np.sin(t) * np.cos(t)) / (1 + np.sin(t)**2)
omega = t
 
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
 
ax.plot(x, y, omega, color='steelblue', linewidth=2)
 
# Proyección sombra en z=0
ax.plot(x, y, np.zeros_like(t), 'k--', alpha=0.3, linewidth=1.2)
 
ax.set_xlabel('$x_1$', fontsize=11)
ax.set_ylabel('$x_2$', fontsize=11)
ax.set_zlabel('$\\omega$', fontsize=11)

y_min, y_max = np.min(y), np.max(y)
ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max), 1))

ax.view_init(elev=25, azim=-60)
 
plt.tight_layout()

# Mostrar en pantalla en vez de guardar
plt.show()