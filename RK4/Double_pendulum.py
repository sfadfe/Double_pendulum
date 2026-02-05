import numpy as np

class Double_pendulum:
    def __init__(self, m1, m2, L1, L2, initial_state, g=9.81):
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = g
        # state: [theta1, omega1, theta2, omega2]
        self.state = np.array(initial_state, dtype=np.float64)

    def equations(self, state):
        t1, w1, t2, w2 = state
        

        delta = t1 - t2
        s_delta = np.sin(delta)
        c_delta = np.cos(delta)
        
        den = 2 * self.m1 + self.m2 - self.m2 * np.cos(2 * delta)
        
        # theta1의 각가속도 (domega1_dt)
        num1 = -self.g * (2 * self.m1 + self.m2) * np.sin(t1)
        num1 -= self.m2 * self.g * np.sin(t1 - 2 * t2)
        num1 -= 2 * s_delta * self.m2 * (w2**2 * self.L2 + w1**2 * self.L1 * c_delta)
        domega1_dt = num1 / (self.L1 * den)
        
        # theta2의 각가속도 (domega2_dt)
        num2 = 2 * s_delta
        num2 *= (w1**2 * self.L1 * (self.m1 + self.m2) + 
                 self.g * (self.m1 + self.m2) * np.cos(t1) + 
                 w2**2 * self.L2 * self.m2 * c_delta)
        domega2_dt = num2 / (self.L2 * den)
        
        # 반환: [d_theta1, d_omega1, d_theta2, d_omega2]: 초기 상태
        return np.array([w1, domega1_dt, w2, domega2_dt])

    def RK4(self, dt):
        k1 = self.equations(self.state)
        k2 = self.equations(self.state + 0.5 * dt * k1)
        k3 = self.equations(self.state + 0.5 * dt * k2)
        k4 = self.equations(self.state + dt * k3)

        self.state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)