import torch
import numpy as np
import time 
from tqdm import tqdm
from PINNs_torch import PINNs, get_device
import os
from torch.utils.data import TensorDataset, DataLoader
import gc

# [설정]
torch.set_float32_matmul_precision('high')
device = get_device()

current_time = time.localtime()
timee = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)    

epochs = 1000000
batch_size = 131072
lr = 1e-4 
sch_pat = 25
lambda_physics = 0
weight_decay = 1e-4
hidden_sizes = [256, 256, 128, 128, 128, 128]
IsPhysicslambda = False
IsLambdaPhysics1 = False

TargetLossRawPhysics = 300000.0
TargetValLoss = 0.00083
MaxLambda = 0.5
OnLambdaPhysics = 1e-9
PhysicsOnTie = 100
tie = 0

filepath = "data/learning_data/RK4.npy"
save_dir_base = f"model_{timee}"
scaler_dir = f"scaler_{timee}.npy"
aa = f'data/models/{timee}'
os.makedirs(aa, exist_ok=False)

print("Loading Data...")
raw_data = np.load(filepath)
X = raw_data[:, :-1, :]
T = raw_data[:, 1:, :4]
X_flat = X.reshape(-1, 12)   
T_flat = T.reshape(-1, 4)
X_mean = X_flat.mean(axis=0)
X_std = X_flat.std(axis=0) + 1e-8
X_flat_norm = (X_flat - X_mean) / X_std
T_mean = T_flat.mean(axis=0)
T_std = T_flat.std(axis=0) + 1e-8
T_flat_norm = (T_flat - T_mean) / T_std

np.save(os.path.join(aa, scaler_dir), {'X_mean': X_mean, 'X_std': X_std, 'T_mean': T_mean, 'T_std': T_std})

total_len = X_flat_norm.shape[0]
split_ratio = 0.8
split_idx = int(total_len * split_ratio)
X_train_np = X_flat_norm[:split_idx]
T_train_np = T_flat_norm[:split_idx]
X_val_np = X_flat_norm[split_idx:]
T_val_np = T_flat_norm[split_idx:]
X_train = torch.tensor(X_train_np, dtype=torch.float32)
T_train = torch.tensor(T_train_np, dtype=torch.float32)
X_val = torch.tensor(X_val_np, dtype=torch.float32)
T_val = torch.tensor(T_val_np, dtype=torch.float32)
del raw_data, X, T, X_flat, T_flat, X_flat_norm, T_flat_norm, X_train_np, T_train_np, X_val_np, T_val_np
gc.collect()

train_dataset = TensorDataset(X_train, T_train)
val_dataset = TensorDataset(X_val, T_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True
)

T_mean_gpu = torch.tensor(T_mean, device=device, dtype=torch.float32)
T_std_gpu = torch.tensor(T_std, device=device, dtype=torch.float32)

model = PINNs(input_size=12, hidden_sizes=hidden_sizes, output_size=4, X_mean=X_mean, X_std=X_std)
model.register_buffer('T_mean', T_mean_gpu)
model.register_buffer('T_std', T_std_gpu)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=sch_pat, min_lr=1e-9)

best_val_loss = float('inf')
best_val_state_dict = None
best_physics_loss = float('inf')
best_physics_state_dict = None

val_loss = 1.0
LossPhysicsRaw = torch.tensor(0.0)
loss_data = torch.tensor(0.0)
val_check_batches = 50

pbar = tqdm(range(epochs), desc="Training", mininterval=1.0, ascii=True, smoothing=0.1)

print("Start Training (Float32 Precision)...")

try:
    for epoch in pbar:
        model.train()
        train_loss_sum = 0.0
        iteration = 0
        
        batch_pbar = tqdm(train_loader, desc=f"Ep {epoch}", leave=False)

        for x_batch, t_batch in batch_pbar:
            x_batch = x_batch.to(device, non_blocking=True)
            t_batch = t_batch.to(device, non_blocking=True)

            if epoch > 2000:
                noise = torch.randn_like(x_batch) * 0.001
                x_batch = x_batch + noise

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss_data = torch.mean((y_pred - t_batch) ** 2)
            
            y_pred_real = y_pred * model.T_std + model.T_mean
            t_true_real = t_batch * model.T_std + model.T_mean
            x_real = x_batch * model.X_std + model.X_mean
            physics_params = x_real[:, 4:8]
            
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
            
            batch_pbar.set_postfix({'L': f'{loss.item():.5f}'})

        avg_train_loss = train_loss_sum / iteration

        if epoch % 20 == 0:
            model.eval()
            val_loss_sum = 0.0
            val_iter = 0
            
            with torch.no_grad():
                for i, (x_val_batch, t_val_batch) in enumerate(val_loader):
                    if i >= val_check_batches: break
                    
                    x_val_batch = x_val_batch.to(device, non_blocking=True)
                    t_val_batch = t_val_batch.to(device, non_blocking=True)
                    
                    y_val_pred = model(x_val_batch)
                    val_loss_sum += torch.mean((y_val_pred - t_val_batch) ** 2).item()
                    val_iter += 1
                
                if val_iter > 0:
                    val_loss = val_loss_sum / val_iter
                else:
                    val_loss = avg_train_loss

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
            if (val_loss < 0.0023) and (current_physics_loss_val < best_physics_loss):
                best_physics_loss = current_physics_loss_val
                best_physics_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix({
            'Train': f'{avg_train_loss:.5f}', 
            'Val': f'{val_loss:.5f}', 
            'B_Val': f'{best_val_loss:.5f}', 
            'B_Phy': f'{best_physics_loss:.1f}', 
            'D': f'{loss_data.item():.5f}',
            'P': f'{(LossPhysicsRaw * lambda_physics).item():.5f}',
            'LR' : f"{optimizer.param_groups[0]['lr']:.2e}",
        })
        
        if (epoch + 1) % 20 == 0:
            pbar.write(f"Epoch {epoch+1} | Val: {val_loss:.6f} | LR = {optimizer.param_groups[0]['lr']:.5e}")

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