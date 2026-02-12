import numpy as np
import secrets

def generate_and_save_states(filename, n_samples):
    theta = np.random.uniform(-np.pi, np.pi, (n_samples, 2))
    omega = np.random.uniform(-10, 10, (n_samples, 2))
    
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(n_samples):
            line = f"{theta[i, 0]},{omega[i, 0]},{theta[i, 1]},{omega[i, 1]}\n"
            f.write(line)

num_samples = 50000

file_route = "RK4"
file_name = "initial_states.txt"

generate_and_save_states(f'{file_route}/{file_name}', num_samples)
print(num_samples)