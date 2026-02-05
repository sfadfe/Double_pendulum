import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from PINNs_torch import PINNs, get_device
import os

time_stamp = "2026-02-04_22-06-36"
folder_path = f"PINN/modelws/{time_stamp}" 
scaler_path = f"{folder_path}/scaler_{time_stamp}.npy"
data_path = "PINN/learning_data/RK4.npy"

files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
physics_models = [f for f in files if "physics" in f]
best_file = sorted(physics_models)[0] if physics_models else sorted(files)[0]
model_path = os.path.join(folder_path, best_file)

future_steps = 1000
dt = 0.01
energy_tolerance_percent = 5.0 

m1, m2 = 1.0, 1.0
l1, l2 = 1.0, 1.0
g = 9.81

device = get_device()

scaler = np.load(scaler_path, allow_pickle=True).item()
model = PINNs(input_size=8, hidden_sizes=[256, 256, 128, 128, 128, 128], output_size=4, 
              X_mean=scaler['X_mean'], X_std=scaler['X_std'])

T_mean_tensor = torch.tensor(scaler['T_mean'], dtype=torch.float32)
T_std_tensor = torch.tensor(scaler['T_std'], dtype=torch.float32)
model.register_buffer('T_mean', T_mean_tensor)
model.register_buffer('T_std', T_std_tensor)
model = model.to(device)

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint['model_state_dict']
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

def get_accelerations(th1, th2, w1, w2):
    delta = th1 - th2
    den1 = (2 * m1 + m2) - m2 * np.cos(2 * delta)
    num1 = -g * (2 * m1 + m2) * np.sin(th1) - m2 * g * np.sin(th1 - 2 * th2) \
           - 2 * np.sin(delta) * m2 * ((w2 ** 2) * l2 + (w1 ** 2) * l1 * np.cos(delta))
    acc1 = num1 / (l1 * den1)
    num2 = 2 * np.sin(delta) * ((w1 ** 2) * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(th1) \
           + (w2 ** 2) * l2 * m2 * np.cos(delta))
    acc2 = num2 / (l2 * den1)
    return acc1, acc2

def get_energy(th1, th2, w1, w2):
    T = 0.5 * m1 * (l1 * w1)**2 + 0.5 * m2 * ((l1 * w1)**2 + (l2 * w2)**2 + 
        2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2))
    V = -m1 * g * l1 * np.cos(th1) - m2 * g * (l1 * np.cos(th1) + l2 * np.cos(th2))
    return T + V

raw_data = np.load(data_path)
X_init = raw_data[0, 0, :].reshape(1, 8)
current_input = torch.tensor((X_init - scaler['X_mean']) / scaler['X_std'], 
                             dtype=torch.float32, device=device)

energies = []
th1_list, th2_list = [], []

with torch.no_grad():
    pred = model(current_input)
    for _ in range(future_steps):
        pred_real = pred * model.T_std + model.T_mean
        th1, th2 = pred_real[0, 0].item(), pred_real[0, 1].item()
        w1, w2 = pred_real[0, 2].item(), pred_real[0, 3].item()
        th1_list.append(th1)
        th2_list.append(th2)
        E = get_energy(th1, th2, w1, w2)
        energies.append(E)
        acc1, acc2 = get_accelerations(th1, th2, w1, w2)
        next_input_raw = torch.tensor([[np.sin(th1), np.cos(th1), np.sin(th2), np.cos(th2), w1, w2, acc1, acc2]], 
                                     dtype=torch.float32, device=device)
        current_input = (next_input_raw - model.X_mean) / model.X_std
        pred = model(current_input)

initial_energy = energies[0]
valid_steps = future_steps
for step, E_curr in enumerate(energies):
    if abs(E_curr - initial_energy) / abs(initial_energy) * 100 > energy_tolerance_percent:
        valid_steps = step
        break

print(f"Model: {best_file}")
print(f"Energy Conservation Time: {valid_steps * dt:.2f} sec")

x1, y1 = l1 * np.sin(th1_list), -l1 * np.cos(th1_list)
x2, y2 = x1 + l2 * np.sin(th2_list), y1 - l2 * np.cos(th2_list)

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1.5])

ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(-(l1+l2)-0.5, (l1+l2)+0.5); ax1.set_ylim(-(l1+l2)-0.5, (l1+l2)+0.5)
ax1.set_aspect('equal'); ax1.grid()
line, = ax1.plot([], [], 'o-', lw=2, color='black')
trace, = ax1.plot([], [], '-', lw=1, color='gray', alpha=0.5)
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(0, future_steps)
ax2.set_ylim(min(energies)-0.5, max(energies)+0.5); ax2.grid()
ax2.axhline(initial_energy, color='green', linestyle='--', alpha=0.5)
if valid_steps < future_steps:
    ax2.axvline(valid_steps, color='red', linestyle='--')
energy_line, = ax2.plot([], [], 'r-', lw=2)

history_x, history_y = [], []

def init():
    line.set_data([], []); trace.set_data([], []); time_text.set_text(''); energy_line.set_data([], [])
    return line, trace, time_text, energy_line

def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    if frame > 0:
        history_x.append(x2[frame]); history_y.append(y2[frame])
        if len(history_x) > 100: history_x.pop(0); history_y.pop(0)
        trace.set_data(history_x, history_y)
    time_text.set_text(f'Time: {frame*dt:.2f}s')
    energy_line.set_data(range(frame), energies[:frame])
    return line, trace, time_text, energy_line

ani = animation.FuncAnimation(fig, update, frames=len(th1_list), init_func=init, interval=20, blit=True)
ani.save(os.path.join(folder_path, "energy_check.mp4"), writer='ffmpeg', fps=60)
plt.show()