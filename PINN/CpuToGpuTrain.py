import torch
import numpy as np
import time
from tqdm import tqdm
from PINNs_torch import PINNs, get_device
import os

torch.set_float32_matmul_precision('high')
device = get_device()

IsPhysicslambda = False
IsLambdaPhysics1 = False
current_time = time.localtime()
timee = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)    

epochs = 1000000
batch_size = 65536
lr = 5e-4
sch_pat = 190
lambda_physics = 0
weight_decay = 1e-4
hidden_sizes = [256, 256, 128, 128, 128, 128]

TargetLossRawPhysics = 300000.0
TargetValLoss = 0.0005
MaxLambda = 0.5        

OnLambdaPhysics = 1e-9
PhysicsOnTie = 100
tie = 0

filepath = "PINN/learning_data/RK4.npy"
save_dir_base = f"model_{timee}"
scaler_dir = f"scaler_{timee}.npy"
aa = f'PINN/models/{timee}'
os.makedirs(aa, exist_ok=True)

raw_data = np.load(filepath)

X = raw_data[:, :-1, :] 
T = raw_data[:, 1:, :4]

X_flat = X.reshape(-1, 8)
T_flat = T.reshape(-1, 4)

X_mean = X_flat.mean(axis=0)
X_std = X_flat.std(axis=0) + 1e-8
X_flat_norm = (X_flat - X_mean) / X_std

T_mean = T_flat.mean(axis=0)
T_std = T_flat.std(axis=0) + 1e-8
T_flat_norm = (T_flat - T_mean) / T_std

np.save(os.path.join(aa, scaler_dir), {'X_mean': X_mean, 'X_std': X_std, 'T_mean': T_mean, 'T_std': T_std})

T_tensor = torch.tensor(T_flat_norm, dtype=torch.float32, device='cpu').pin_memory()
X_tensor = torch.tensor(X_flat_norm, dtype=torch.float32, device='cpu').pin_memory()

total_len = X_tensor.shape[0]
split_ratio = 0.8
split_idx = int(total_len * split_ratio)

X_train = X_tensor[:split_idx]
T_train = T_tensor[:split_idx]
X_val = X_tensor[split_idx:]
T_val = T_tensor[split_idx:]

num_train_data = X_train.shape[0]
num_val_data = X_val.shape[0]

T_mean_gpu = torch.tensor(T_mean, device=device, dtype=torch.float32)
T_std_gpu = torch.tensor(T_std, device=device, dtype=torch.float32)

model = PINNs(input_size=8, hidden_sizes=hidden_sizes, output_size=4, X_mean=X_mean, X_std=X_std)
model.register_buffer('T_mean', T_mean_gpu)
model.register_buffer('T_std', T_std_gpu)
model = model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=sch_pat, min_lr=1e-9)

best_val_loss = float('inf')
best_val_state_dict = None

best_physics_loss = float('inf')
best_physics_state_dict = None

val_loss = 1.0
LossPhysicsRaw = torch.tensor(0.0)
loss_data = torch.tensor(0.0)

val_check_batches = 5

pbar = tqdm(range(epochs), desc="Training", mininterval=1.0, ascii=True, smoothing=0.1)

try:
    for epoch in pbar:
        model.train()
        
        curr_indices = torch.randperm(num_train_data, device='cpu')
        train_loss_sum = 0.0
        iteration = 0
        
        for i in range(0, num_train_data, batch_size):
            batch_idx = curr_indices[i : i + batch_size]
            x_batch = X_train[batch_idx].to(device, non_blocking=True)
            t_batch = T_train[batch_idx].to(device, non_blocking=True)

            # [추가] 노이즈 주입 (Robustness 강화)
            if epoch > 2000:
                noise = torch.randn_like(x_batch) * 0.001
                x_batch = x_batch + noise

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss_data = torch.mean((y_pred - t_batch) ** 2)
            
            y_pred_real = y_pred * model.T_std + model.T_mean
            t_true_real = t_batch * model.T_std + model.T_mean
            x_real = x_batch * model.X_std + model.X_mean
            physics_params = x_real[:, 4:]
            
            E_true = model.get_energy(t_true_real, physics_params)
            E_pred = model.get_energy(y_pred_real, physics_params)
            
            LossPhysicsRaw = torch.mean((E_pred - E_true) ** 2)
            LossPhysics = lambda_physics * LossPhysicsRaw
            loss = loss_data + LossPhysics
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            iteration += 1

        avg_train_loss = train_loss_sum / iteration

        if epoch % 20 == 0:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                val_indices = torch.randperm(num_val_data, device='cpu')[:batch_size * val_check_batches]
                
                for i in range(0, len(val_indices), batch_size):
                    batch_idx = val_indices[i : i + batch_size]
                    x_val_batch = X_val[batch_idx].to(device, non_blocking=True)
                    t_val_batch = T_val[batch_idx].to(device, non_blocking=True)
                    
                    y_val_pred = model(x_val_batch)
                    val_loss_sum += torch.mean((y_val_pred - t_val_batch) ** 2).item()
                
                val_loss = val_loss_sum / val_check_batches

            scheduler.step(val_loss)
            
            if IsPhysicslambda == False:
                if val_loss < TargetValLoss:
                    tie += 1
                else:
                    tie = 0

                if tie >= PhysicsOnTie/20:
                    IsPhysicslambda = True
                    lambda_physics = OnLambdaPhysics
                    
                    pbar.write(f"\nPhysics Activated at epoch {epoch+1}")
                    pbar.write(f"-> Start Lambda: {lambda_physics:.6e}")

                    tie = 0
                    OnLambdaPhysics *= 1.5
                    TargetLossRawPhysics *= 0.68
                    PhysicsOnTie += 320

            else:
                if (LossPhysicsRaw < TargetLossRawPhysics) and (val_loss < TargetValLoss):
                    tie += 1
                else:
                    tie = 0

                if tie >= PhysicsOnTie/20:  
                    lambda_physics = min(OnLambdaPhysics, MaxLambda)
                    IsLambdaPhysics1 = True if lambda_physics >= 1.0 else False
                    
                    pbar.write(f"\nEpoch {epoch+1}: P < {TargetLossRawPhysics:.0f}, Val < {TargetValLoss}")
                    pbar.write(f"-> New Lambda: {lambda_physics:.6e}")

                    tie = 0
                    OnLambdaPhysics *= 2
                    TargetLossRawPhysics *= 0.8
                    PhysicsOnTie += 40

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            current_physics_loss_val = LossPhysicsRaw.item()
            if (val_loss < 0.001) and (current_physics_loss_val < best_physics_loss):
                best_physics_loss = current_physics_loss_val
                best_physics_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix({
            'Train': f'{avg_train_loss:.5f}', 
            'Val': f'{val_loss:.5f}', 
            'B_Val': f'{best_val_loss:.5f}', 
            'B_Phy': f'{best_physics_loss:.1f}', 
            'D': f'{loss_data.item():.5f}',
            'P': f'{(LossPhysicsRaw * lambda_physics).item():.5f}'
        })

        if (epoch + 1) % 100 == 0:
            pbar.write(f"Epoch {epoch+1}/{epochs} | Val: {val_loss:.6f}")

except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected. Saving models...")

finally:
    if best_val_state_dict is not None:
        save_path_val = os.path.join(aa, f"{save_dir_base}_best_val_{best_val_loss:.6f}.pt")
        save_content_val = {
            'model_state_dict': best_val_state_dict,
            'epoch': epoch,
            'best_loss': best_val_loss,
            'config': {'lambda_physics': lambda_physics, 'note': 'Best Validation Loss Model'}
        }
        torch.save(save_content_val, save_path_val)
        print(f"Saved Best Val Model: {save_path_val}")

    if best_physics_state_dict is not None:
        save_path_phys = os.path.join(aa, f"{save_dir_base}_best_physics_{best_physics_loss:.1f}.pt")
        save_content_phys = {
            'model_state_dict': best_physics_state_dict,
            'epoch': epoch,
            'best_physics_loss': best_physics_loss,
            'config': {'lambda_physics': lambda_physics, 'note': 'Best Physics Loss Model'}
        }
        torch.save(save_content_phys, save_path_phys)
        print(f"Saved Best Physics Model: {save_path_phys}")

    with open(os.path.join(aa, 'info.txt'), 'w') as f:
        f.write(f"Best Val Loss: {best_val_loss:.6f}\n")
        f.write(f"Best Physics Raw Loss: {best_physics_loss}\n")
        f.write(f"Epochs Trained: {epoch+1}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Final Lambda: {lambda_physics}\n")
        f.write(f"Learning Rate: {lr}\n")