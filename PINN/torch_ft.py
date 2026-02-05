## fine-tuning(파인 튜닝)
import torch
import numpy as np
from tqdm import tqdm
from PINNs_torch import PINNs, get_device
import os

device = get_device()

a = "2026-02-04_22-06-36"
folder_path = f"PINN/models/{a}"

model_path = os.path.join(folder_path, f"model_{a}.pt")
scaler_path = os.path.join(folder_path, f"scaler_{a}.npy")
output_path = os.path.join(folder_path, f"model_finetuned.pt")
dataset = "PINN/learning_data/RK4.npy"

ScalerDate = np.load(scaler_path, allow_pickle=True).item()

X_mean = ScalerDate['X_mean']
X_std = ScalerDate['X_std']
T_mean = ScalerDate['T_mean']
T_std = ScalerDate['T_std']

raw_data = np.load(dataset)

X = raw_data[:, :-1, :]
T = raw_data[:, 1:, :4]

X_flat = X.reshape(-1, 8)
T_flat = T.reshape(-1, 4)

X_flat_norm = (X_flat - X_mean) / X_std
T_flat_norm = (T_flat - T_mean) / T_std

X_tensor = torch.tensor(X_flat_norm, dtype=torch.float32, device=device)
T_tensor = torch.tensor(T_flat_norm, dtype=torch.float32, device=device)

total_len = X_tensor.shape[0]
split_ratio = 0.85
split_idx = int(total_len * split_ratio)

X_train = X_tensor[:split_idx]
T_train = T_tensor[:split_idx]
X_val = X_tensor[split_idx:]
T_val = T_tensor[split_idx:]

num_train_data = X_train.shape[0]

T_mean_gpu = torch.tensor(T_mean, device=device, dtype=torch.float32)
T_std_gpu = torch.tensor(T_std, device=device, dtype=torch.float32)

model = PINNs(input_size=8, hidden_sizes=[256, 256, 128, 128, 128, 128], output_size=4, X_mean=X_mean, X_std=X_std)
model.register_buffer('T_mean', T_mean_gpu)
model.register_buffer('T_std', T_std_gpu)
model = model.to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


lr = 1e-5
epochs = 500000
batch_size = 32768

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.9, patience=2000, min_lr=1e-8
)

best_loss = checkpoint['best_loss']
best_state_dict = None
lambda_physics = 1e-4

pbar = tqdm(range(epochs), desc="Fine-Tuning", mininterval=1.0, ascii=True, smoothing=0.1)

try:
    for epoch in pbar:
        model.train()
        
        curr_indices = torch.randperm(num_train_data, device=device)
        train_loss_sum = 0.0
        iteration = 0
        
        for i in range(0, num_train_data, batch_size):
            batch_idx = curr_indices[i : i + batch_size]
            x_batch = X_train[batch_idx]
            t_batch = T_train[batch_idx]

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss_data = torch.mean((y_pred - t_batch) ** 2)
            
            # 물리 오차 계산
            y_pred_real = y_pred * model.T_std + model.T_mean
            t_true_real = t_batch * model.T_std + model.T_mean
            x_real = x_batch * model.X_std + model.X_mean
            physics_params = x_real[:, 4:]
            
            E_true = model.get_energy(t_true_real, physics_params)
            E_pred = model.get_energy(y_pred_real, physics_params)
            
            loss_physics_raw = torch.mean((E_pred - E_true) ** 2)
            loss = loss_data + lambda_physics * loss_physics_raw
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            iteration += 1

        avg_train_loss = train_loss_sum / iteration

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = torch.mean((y_val_pred - T_val) ** 2)

        scheduler.step(val_loss)

        # 베스트 모델 갱신 (이전 학습의 기록보다 좋아져야 갱신됨)
        if val_loss < best_loss:
            best_loss = val_loss.item()
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix({
            'Train': f'{avg_train_loss:.6f}', 
            'Val': f'{val_loss.item():.6f}', 
            'Best': f'{best_loss:.6f}',
            'P': f'{(loss_physics_raw * lambda_physics).item():.6f}'
        })

        if (epoch + 1) % 100 == 0:
            pbar.write(f"Fine-Tune Epoch {epoch+1}/{epochs} | Val: {val_loss:.7f}")

except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected. Saving best model...")

finally:
    if best_state_dict is not None:
        save_content = {
                'model_state_dict': best_state_dict,
                'epoch': epoch,
                'best_loss': best_loss,
                'config': {
                'split_ratio': split_ratio,
                'batch_size': batch_size,
                'lambda_physics': lambda_physics,
                'learning_rate': lr,
                'base_model': model_path
                }
            }
        
        torch.save(save_content, output_path)

        with open(os.path.join(folder_path, 'finetune_info.txt'), 'w') as f:
            f.write(f"Base Model: {model_path}\n")
            f.write(f"Best Val Loss: {best_loss:.7f}\n")
            f.write(f"Epochs Trained: {epoch+1}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Lambda Physics: {lambda_physics}\n")
            f.write(f"Learning Rate: {lr}\n")

        print(f"Final Best Val Loss: {best_loss:.7f}")

    else:
        print("fine-tuning failed") 