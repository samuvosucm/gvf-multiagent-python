import numpy as np
import matplotlib.pyplot as plt

a = 5
t = np.linspace(0, 2*np.pi, 1000)

x = (a * np.cos(t)) / (1 + np.sin(t)**2)
y = (a * np.sin(t) * np.cos(t)) / (1 + np.sin(t)**2)

plt.figure(figsize=(6,6))
plt.plot(x, y, 'k', linewidth=1)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False, handlelength=0)
margin = 2
plt.xlim(np.min(x) - margin, np.max(x) + margin)
plt.ylim(np.min(y) - margin, np.max(y) + margin)
plt.grid(True)

plt.gca().set_aspect('equal')
plt.show()