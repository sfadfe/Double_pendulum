import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import Double_pendulum as Dp
import time

L1, L2 = 1.0, 1.0
initial_state =[-3.02848877,  6.99445852,  0.79264971, -0.21103231]
dp = Dp.Double_pendulum(m1=1.0, m2=1.0, L1=L1, L2   =L2, initial_state=initial_state)

dt = 0.03
t_max = 10
steps = int(t_max / dt)

x1_data, y1_data = [], []
x2_data, y2_data = [], []

for i in range(steps):
    dp.RK4(dt)
    
    theta1 = dp.state[0]
    theta2 = dp.state[2]
    
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    
    x1_data.append(x1)
    y1_data.append(y1)
    x2_data.append(x2)
    y2_data.append(y2)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-(L1+L2+0.5), (L1+L2+0.5))
ax.set_ylim(-(L1+L2+0.5), (L1+L2+0.5))
ax.set_aspect('equal')
ax.grid(True)

line, = ax.plot([], [], 'o-', lw=2, color='black') 
trace, = ax.plot([], [], '-', lw=1, color='red', alpha=0.5)

def update(frame):
    x1, y1 = x1_data[frame], y1_data[frame]
    x2, y2 = x2_data[frame], y2_data[frame]
    
    line.set_data([0, x1, x2], [0, y1, y2])
    trace.set_data(x2_data[:frame], y2_data[:frame])
    
    return line, trace

ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True)

writer = PillowWriter(fps=20) 

timestamp = int(time.time())
filename = f"double_pendulum_{timestamp}.gif"
filepath = "Double_pendulum/RK4_ver/results"

ani.save(f"{filepath}/{filename}", writer=writer)

plt.close()