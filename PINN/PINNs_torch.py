import torch
import torch.nn as nn

class PINNs(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, X_mean, X_std, ):
        super(PINNs, self).__init__()
        
        self.register_buffer('X_mean', torch.tensor(X_mean, dtype=torch.float32))
        self.register_buffer('X_std', torch.tensor(X_std, dtype=torch.float32))
        
        self.g = 9.81
        
        # 레이어 구성
        all_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(all_sizes) - 1):
            layers.append(nn.Linear(all_sizes[i], all_sizes[i+1]))
            # Xavier 초기화 & Bias 0 초기화
            nn.init.xavier_uniform_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            if i < len(all_sizes) - 2:
                layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_energy(self, state, params):
        th1, w1, th2, w2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        m1, m2, L1, L2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

        y1 = -L1 * torch.cos(th1)
        y2 = y1 - L2 * torch.cos(th2)
        V = (m1 + m2) * self.g * y1 - m2 * self.g * L2 * torch.cos(th2) 
        V = -(m1 + m2) * self.g * L1 * torch.cos(th1) - m2 * self.g * L2 * torch.cos(th2)

        v1_sq = (L1 * w1)**2
        v2_sq = (L1 * w1)**2 + (L2 * w2)**2 + \
                2 * L1 * L2 * w1 * w2 * torch.cos(th1 - th2)
        T = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
        
        return T + V

    def compute_loss(self, y_pred, t_true, x_input_norm, lambda_p=0.1):
        
        # 데이터 Loss: 정규화된 값으로
        loss_data = torch.mean((y_pred - t_true) ** 2)
        
        if lambda_p == 0:
            return loss_data

        # 물리 Loss: 실제값으로
        x_real = x_input_norm * self.X_std + self.X_mean
        physics_params = x_real[:, 4:] 

        t_true_real = t_true * self.T_std + self.T_mean
        y_pred_real = y_pred * self.T_std + self.T_mean

        E_true = self.get_energy(t_true_real, physics_params)
        E_pred = self.get_energy(y_pred_real, physics_params)
        
        diff_sq = (E_pred - E_true) ** 2
        scale = E_true ** 2 + 1e-8
        
        loss_physics = torch.mean(diff_sq / scale)
        
        return loss_data + lambda_p * loss_physics

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    elif torch.backends.mps.is_available(): return torch.device('mps')
    else: return torch.device('cpu')