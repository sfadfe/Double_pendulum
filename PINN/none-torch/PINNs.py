import cupy as cp
import numpy as np
from Autograd import Tensor as Tsr

class PINNs:
    def __init__(self, input_size, hidden_sizes, output_size, X_mean, X_std, T_mean, T_std):
        self.g = 9.81
        self.X_mean = cp.array(X_mean, dtype=cp.float32)
        self.X_std = cp.array(X_std, dtype=cp.float32)
        self.T_mean = cp.array(T_mean, dtype=cp.float32)
        self.T_std = cp.array(T_std, dtype=cp.float32)
        
        self.params = {}
        self.hidden_sizes = hidden_sizes
        
        all_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(all_sizes) - 1
        
        for i in range(self.num_layers):
            in_node = all_sizes[i]
            out_node = all_sizes[i+1]

            w_key = 'W' + str(i + 1)
            b_key = 'b' + str(i + 1)
            
            # Xavier Initialization
            scale = cp.sqrt(1.0 / in_node)
            self.params[w_key] = Tsr(cp.random.randn(in_node, out_node) * scale)
            self.params[b_key] = Tsr(cp.zeros(out_node))

    def forward(self, x):
        out = x
        for i in range(1, self.num_layers):            
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            z = out @ W + b
            out = z.tanh()

        last_idx = self.num_layers
        W_last = self.params['W' + str(last_idx)]
        b_last = self.params['b' + str(last_idx)]
        out = out @ W_last + b_last
        return out

    def get_energy(self, state, params):
        # Tensor 슬라이싱
        th1 = state[:, 0]
        w1  = state[:, 1]
        th2 = state[:, 2]
        w2  = state[:, 3]
        
        m1 = params[:, 0]
        m2 = params[:, 1]
        L1 = params[:, 2]
        L2 = params[:, 3]

        # 1. 위치 에너지 계산
        y1 = -L1 * th1.cos()
        y2 = y1 - L2 * th2.cos()
        V = m1 * self.g * y1 + m2 * self.g * y2
        
        # 2. 운동 에너지 계산
        v1_sq = (L1 * w1)**2
        v2_sq = (L1 * w1)**2 + (L2 * w2)**2 + \
                2 * L1 * L2 * w1 * w2 * (th1 - th2).cos()
                
        T = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
        
        return T + V

    def loss(self, x_input, y_pred, t_true):
        batch_size = y_pred.data.shape[0]
        
        loss_data = 0.5 * ((y_pred - t_true) ** 2).sum() / batch_size
        
        x_real = x_input * self.X_std + self.X_mean
        y_real_state = y_pred * self.T_std + self.T_mean
        
        # x_real: [th1, w1, th2, w2, m1, m2, L1, L2]
        params_real = x_real[:, 4:]
        
        E_in = self.get_energy(x_real[:, :4], params_real)
        
        E_pred = self.get_energy(y_real_state, params_real)
        
        loss_physics_raw = 0.5 * ((E_pred - E_in) ** 2).sum() / batch_size
    
        
        lambda_p = 0.1
        
        return loss_data + lambda_p * loss_physics_raw

class AdamW:
    def __init__(self, lr=0.005, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = cp.zeros_like(val)
                self.v[key] = cp.zeros_like(val)

        self.t += 1

        for key in params.keys():
            if key in grads:
                params[key] -= self.lr * self.weight_decay * params[key]

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key]**2)

                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                params[key] -= self.lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)

class AdamW_AutoGrad:
    def __init__(self, lr=0.005, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0


    def update(self, params):


        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = cp.zeros_like(val.data)
                self.v[key] = cp.zeros_like(val.data)

        self.t += 1

        for key in params.keys():

            grads = params[key].grad
            data = params[key].data

            data -= self.lr * self.weight_decay * data

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads**2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            data -= self.lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)

            params[key].grad = cp.zeros_like(data)