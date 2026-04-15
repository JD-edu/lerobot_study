import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
import os

class SO101ReachEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(SO101ReachEnv, self).__init__()
        
        # 1. 모델 로드 (파일 경로 확인 필요)
        xml_path = "./scene.xml" 
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        
        # 2. Action 및 Observation Space 정의
        # Action: 6개의 액추에이터 제어값 (-1.0 ~ 1.0)
        self.nu = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)
        
        # Observation: 관절 위치(qpos) + 관절 속도(qvel) + 목표 좌표(target_xyz)
        # qpos/qvel의 개수는 모델의 DOF에 따라 다를 수 있으므로 동적으로 설정
        obs_shape = self.model.nq + self.model.nv + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def _get_obs(self):
        # 현재 상태 정보 추출
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            self.target_pos
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 시뮬레이션 초기화
        mujoco.mj_resetData(self.model, self.data)
        
        # 랜덤 목표 지점 생성 (로봇 가동 범위 내)
        self.target_pos = np.array([
            np.random.uniform(0.15, 0.35), # X
            np.random.uniform(-0.2, 0.2),  # Y
            np.random.uniform(0.1, 0.4)    # Z
        ])
        
        # 시각화를 위해 목표 지점에 'site'나 'mocap' 바디 위치를 업데이트할 수 있음
        # 여기서는 단순 계산을 위해 내부 변수로만 활용
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Action 적용 (액추에이터 제어)
        self.data.ctrl[:self.nu] = action
        
        # 2. 시뮬레이션 진행
        mujoco.mj_step(self.model, self.data)
        
        # 3. 보상(Reward) 계산
        # End-effector 위치 가져오기 (XML에 'effector'라는 이름의 site가 있다고 가정)
        # 이름이 다르다면 self.model.site('이름').id로 확인 필요
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "effector")
        current_ee_pos = self.data.site_xpos[ee_id]
        
        dist = np.linalg.norm(current_ee_pos - self.target_pos)
        
        # 보상 함수 설계
        reward = -dist # 거리가 멀수록 마이너스
        reward -= 0.01 * np.square(action).sum() # 에너지 효율 페널티
        
        if dist < 0.02: # 목표 도달 성공 보상
            reward += 10.0
            terminated = True
        else:
            terminated = False
            
        truncated = False # 시간 초과 등은 Wrapper에서 처리 가능
        
        return self._get_obs(), reward, terminated, truncated, {}

# --- 메인 학습 실행 ---
if __name__ == "__main__":
    # 1. 환경 생성
    env = SO101ReachEnv()

    # 2. 알고리즘 설정 (PPO)
    # MLP(Multi-Layer Perceptron) 정책 사용
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_so101_logs/")

    # 3. 학습 시작
    print("학습을 시작합니다...")
    model.learn(total_timesteps=200000) # 충분한 학습을 위해 시간 투자 필요

    # 4. 모델 저장
    model.save("so101_reach_model")
    print("학습 완료 및 모델 저장 완료!")

    # 5. 테스트 실행 (시각화)
    obs, _ = env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            if terminated or truncated:
                obs, _ = env.reset()