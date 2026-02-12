import numpy as np
import os
import Double_pendulum as Dp
from tqdm import tqdm
import multiprocessing

def simulate_single_trajectory(args):
    line, dt, steps, SaveInterval = args
    
    # 빈 줄 처리
    if not line.strip():
        return None

    L1 = float(np.random.uniform(0.2, 3.0))
    L2 = float(np.random.uniform(0.2, 3.0))
    m1 = float(np.random.uniform(0.5, 6.5))
    m2 = float(np.random.uniform(0.5, 6.5))

    if np.random.rand() < 0.07: # 질량 비 극단 케이스 만들
        m1, m2 = 0.7, 0.1
        if np.random.rand() < 0.5:
            m1, m2 = m2, m1

    # 초기 상태
    parts = line.replace(',', ' ').split()
    initial_state = list(map(float, parts))

    dp = Dp.Double_pendulum(m1, m2, L1=L1, L2=L2, initial_state=initial_state)
    
    local_trajectory = []

    """
    데이터 행 생성
    [th1, w1, th2, w2] + [m1, m2, L1, L2] + [sin(theta1), cos(theta1), sin(theta2), cos(theta2)]
    """
    def get_data_row(dp_obj, m1, m2, L1, L2):
        s = dp_obj.state.tolist() # [th1, w1, th2, w2]
        theta1, theta2 = s[0], s[2]
        return s + [m1, m2, L1, L2] + [np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2)]

    local_trajectory.append(get_data_row(dp, m1, m2, L1, L2))

    for i in range(1, steps + 1):
        dp.RK4(dt)
        
        if i % SaveInterval == 0: # 저장 주기
            local_trajectory.append(get_data_row(dp, m1, m2, L1, L2))

    # 메모리 효율을 위해 numpy array로 변환해서 반환
    with open(input_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    worker_args = [(line, dt, steps, SaveInterval) for line in lines]

    #multiprocessing을 위한 코어 수 확인 / 사용 할 코어 수 설
    NumCores = multiprocessing.cpu_count()
    UsingCore = max(1, NumCores - 1) 
    
    print(f"Using {UsingCore} core")

    results = []
    
    with multiprocessing.Pool(processes=UsingCore) as pool:
        for res in tqdm(pool.imap(simulate_single_trajectory, worker_args), total=len(lines), desc="Simulating"):
            if res is not None:
                results.append(res)
    print("Stacking data...")
    # 모든 결과 합치기 (데이터 개수, 시간, 12)
    final_data = np.array(results, dtype=np.float32)
    
    save_path = os.path.join(output_dir, "RK4.npy")
    np.save(save_path, final_data)

    print(f"Done... {final_data.shape}")
    return np.array(local_trajectory, dtype=np.float32)

if __name__ == '__main__':
    
    input_path = "RK4/initial_states.txt"
    output_dir = "data/learning_data"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dt = 1e-5
    t_max = 10
    steps = int(t_max / dt)
    SaveInterval = 250 # 250 steps = 0.0025 sec

    with open(input_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    worker_args = [(line, dt, steps, SaveInterval) for line in lines]

    # multiprocessing을 위한 코어 수 확인 / 사용 할 코어 수 설
    NumCores = multiprocessing.cpu_count()
    UsingCore = max(1, NumCores - 1) 
    
    print(f"Using {UsingCore} core")

    results = []
    
    with multiprocessing.Pool(processes=UsingCore) as pool:
        for res in tqdm(pool.imap(simulate_single_trajectory, worker_args), total=len(lines), desc="Simulating"):
            if res is not None:
                results.append(res)
    print("Stacking data...")
    # 모든 결과 합치기 (데이터 개수, 시간, 12)
    final_data = np.array(results, dtype=np.float32)
    
    save_path = os.path.join(output_dir, "RK4.npy")
    np.save(save_path, final_data)

    print(f"Done... {final_data.shape}")