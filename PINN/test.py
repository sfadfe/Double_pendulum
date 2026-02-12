import torch
import numpy as np
import time 
from tqdm import tqdm
from PINNs_torch import PINNs, get_device
import os
from torch.utils.data import TensorDataset, DataLoader
import gc

# [설정]
DEBUG_MODE = True # 테스트 시 True, 본 학습 시 False
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

# 물리 활성화 조건
TargetLossRawPhysics = 300000.0
TargetValLoss = 0.00083 
MaxLambda = 0.5
OnLambdaPhysics = 1e-9
PhysicsOnTie = 100
tie = 0

filepath = "data/learning_data/RK4.npy"
aa = f'data/models/{timee}'
os.makedirs(aa, exist_ok=True)

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리
# ---------------------------------------------------------
print("Loading Data...")
raw_data = np.load(filepath)
X_flat = raw_data[:, :-1, :].reshape(-1, 12)   
T_flat = raw_data[:, 1:, :4].reshape(-1, 4)

X_mean, X_std = X_flat.mean(axis=0), X_flat.std(axis=0) + 1e-8
T_mean, T_std = T_flat.mean(axis=0), T_flat.std(axis=0) + 1e-8
X_flat_norm = (X_flat - X_mean) / X_std
T_flat_norm = (T_flat - T_mean) / T_std

np.save(os.path.join(aa, f"scaler_{timee}.npy"), 
        {'X_mean': X_mean, 'X_std': X_std, 'T_mean': T_mean, 'T_std': T_std})

total_len = X_flat_norm.shape[0]
split_ratio = 0.8

if DEBUG_MODE:
    indices = np.random.choice(total_len, int(total_len * 0.01), replace=False)
    X_subset = X_flat_norm[indices]
    T_subset = T_flat_norm[indices]
else:
    X_subset = X_flat_norm
    T_subset = T_flat_norm

split_idx = int(len(X_subset) * split_ratio)
X_train = torch.tensor(X_subset[:split_idx], dtype=torch.float32)
T_train = torch.tensor(T_subset[:split_idx], dtype=torch.float32)
X_val = torch.tensor(X_subset[split_idx:], dtype=torch.float32)
T_val = torch.tensor(T_subset[split_idx:], dtype=torch.float32)

del raw_data, X_flat, T_flat, X_flat_norm, T_flat_norm, X_subset, T_subset; gc.collect()

train_loader = DataLoader(TensorDataset(X_train, T_train), batch_size=batch_size, 
                          shuffle=True, pin_memory=True, num_workers=0 if DEBUG_MODE else 8)
val_loader = DataLoader(TensorDataset(X_val, T_val), batch_size=batch_size, shuffle=False)

# ---------------------------------------------------------
# 2. 모델 및 옵티마이저 설정
# ---------------------------------------------------------
model = PINNs(input_size=12, hidden_sizes=hidden_sizes, output_size=4, X_mean=X_mean, X_std=X_std).to(device)

model.register_buffer('T_mean', torch.tensor(T_mean, dtype=torch.float32))
model.register_buffer('T_std', torch.tensor(T_std, dtype=torch.float32))

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=sch_pat)

pbar = tqdm(range(epochs), desc="Training", mininterval=1.0)
val_loss = 1.0
LossPhysicsRaw = torch.tensor(0.0)

# ---------------------------------------------------------
# 3. 학습 루프
# ---------------------------------------------------------
try:
    for epoch in pbar:
        model.train()
        train_loss_sum = 0.0
        
        for x_batch, t_batch in train_loader:
            x_batch, t_batch = x_batch.to(device), t_batch.to(device)
            optimizer.zero_grad()
            
            # Forward (순수 float32)
            y_pred = model(x_batch)
            loss_data = torch.mean((y_pred - t_batch) ** 2)
            
            # Physics Loss (Denormalization 필수)
            x_real = x_batch * model.X_std + model.X_mean
            y_pred_real = y_pred * model.T_std + model.T_mean
            t_true_real = t_batch * model.T_std + model.T_mean
            
            E_pred = model.get_energy(y_pred_real, x_real[:, 4:8])
            E_true = model.get_energy(t_true_real, x_real[:, 4:8])
            
            LossPhysicsRaw = torch.mean((E_pred - E_true) ** 2)
            loss = loss_data + (lambda_physics * LossPhysicsRaw)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item()

        # 검증 및 로직 체크 (5 에폭마다)
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                # 전체 검증 데이터 중 일부만 사용하여 속도 유지
                x_v, t_v = X_val[:batch_size].to(device), T_val[:batch_size].to(device)
                val_loss = torch.mean((model(x_v) - t_v) ** 2).item()
            
            scheduler.step(val_loss)

            # Physics Activation 로직
            if not IsPhysicslambda:
                if val_loss < TargetValLoss: tie += 1
                else: tie = 0
                if tie >= (PhysicsOnTie // 5): # 주기에 맞춰 조정
                    IsPhysicslambda = True
                    lambda_physics = OnLambdaPhysics
                    pbar.write(f"\n[EVENT] Physics Activated at Ep {epoch} | Val: {val_loss:.6f}")
                    tie = 0
            else:
                if (LossPhysicsRaw < TargetLossRawPhysics) and (val_loss < TargetValLoss): tie += 1
                else: tie = 0
                if tie >= (PhysicsOnTie // 5):
                    lambda_physics = min(OnLambdaPhysics * 2, MaxLambda)
                    OnLambdaPhysics = lambda_physics
                    pbar.write(f"\n[EVENT] Lambda Increased: {lambda_physics:.2e}")
                    tie = 0
                    TargetLossRawPhysics *= 0.8

        pbar.set_postfix({
            'D': f'{loss_data.item():.5f}',
            'P_Raw': f'{LossPhysicsRaw.item():.1e}',
            'L_Phy': f'{lambda_physics:.1e}',
            'Val': f'{val_loss:.5f}',
            'Tie': tie
        })

except KeyboardInterrupt:
    print("\nTr     aining interrupted by user.")