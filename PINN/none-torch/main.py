from PINNs import PINNs, AdamW_AutoGrad
import cupy as cp
import numpy as np
import os
from tqdm import tqdm
import time
from Autograd import Tensor as Tsr

current_time = time.localtime()
timee = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)

epochs = 1000000

filepath = "Double_pendulum/ANN_ver/learning_data/RK4.npy"
save_dir = f"Double_pendulum/ANN_ver/models/model_{timee}.npy"
scaler_dir = f"Double_pendulum/ANN_ver/models/scaler_{timee}.npy"

raw_data = np.load(filepath)
    
X = raw_data[:, :-1, :] 
T = raw_data[:, 1:, :4] 

X_flat = X.reshape(-1, 8)
T_flat = T.reshape(-1, 4)

X_mean = X_flat.mean(axis=0)
X_std = X_flat.std(axis=0) + 1e-8
X_flat = (X_flat - X_mean) / X_std

T_mean = T_flat.mean(axis=0)
T_std = T_flat.std(axis=0) + 1e-8
T_flat = (T_flat - T_mean) / T_std

# 스케일러 저장
np.save(scaler_dir, {'X_mean': X_mean, 'X_std': X_std, 'T_mean': T_mean, 'T_std': T_std})
    
X_train = cp.array(X_flat, dtype=cp.float32)
T_train = cp.array(T_flat, dtype=cp.float32)

num_data = X_train.shape[0]
batch_size = 16384

model = PINNs(
    input_size=8, 
    hidden_sizes=[256, 256, 256, 256, 256, 256, 256], 
    output_size=4,
    X_mean=X_mean, X_std=X_std,
    T_mean=T_mean, T_std=T_std
)

optimizer = AdamW_AutoGrad(lr=0.005, weight_decay=1e-4)

best_loss = float('inf')
best_params_in_ram = None

pbar = tqdm(range(epochs), desc="Training", mininterval=1.0, ascii=True, smoothing=0.1)

try:
    for epoch in pbar:
        indices = cp.random.permutation(num_data)
        epoch_loss = 0.0
        iteration = 0

        for i in range(0, num_data, batch_size):
            batch_idx = indices[i : i + batch_size]
            
            x_batch = Tsr(X_train[batch_idx])
            t_batch = Tsr(T_train[batch_idx])

            y_pred = model.forward(x_batch)
            
            loss = model.loss(x_batch, y_pred, t_batch)
            
            loss.backward()
            optimizer.update(model.params)
            
            epoch_loss += loss.data.item()
            iteration += 1

        avg_loss = epoch_loss / iteration

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params_in_ram = {k: v.data.get() for k, v in model.params.items()} 

        pbar.set_postfix({'Loss': f'{avg_loss:.6f}', 'Best': f'{best_loss:.6f}'})

        if (epoch + 1) % 100 == 0:
            pbar.write(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.8f} | Best: {best_loss:.8f}")

except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected. Saving best model...")

finally:
    if best_params_in_ram is not None:
        np.save(save_dir, best_params_in_ram)
        print(f"Model saved to {save_dir}")
        print(f"Best Loss: {best_loss:.8f}")