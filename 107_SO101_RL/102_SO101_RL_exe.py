import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
import time
import os

class SO101ReachEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(SO101ReachEnv, self).__init__()
        
        # 1. 모델 로드 (파일 경로 확인 필요)
        xml_path = "scene.xml" 
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


def run_test():
    # 1. 환경 및 모델 로드
    env = SO101ReachEnv()
    
    # 학습 시 저장했던 파일명 (확장자 .zip은 생략 가능)
    model_path = "so101_reach_model"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"에러: {model_path}.zip 파일을 찾을 수 없습니다. 먼저 학습을 완료해 주세요.")
        return

    model = PPO.load(model_path)
    print("모델 로드 완료! 테스트를 시작합니다.")

    # 2. MuJoCo 수동 렌더링 루프
    obs, _ = env.reset()
    
    # launch_passive를 사용하여 시뮬레이션 창을 띄웁니다.
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # 무한 루프 또는 특정 횟수 반복
        while viewer.is_running():
            step_start = time.time()

            # 모델로부터 최적의 행동(Action) 예측
            # deterministic=True: 무작위성을 배제하고 가장 확률 높은 행동만 선택
            action, _states = model.predict(obs, deterministic=True)

            # 환경에 행동 적용
            obs, reward, terminated, truncated, info = env.step(action)

            # 뷰어 갱신
            viewer.sync()

            # 목표 도달 시 잠시 대기 후 리셋 (시각적 확인을 위해)
            if terminated or truncated:
                print("목표 도달! 리셋합니다.")
                time.sleep(1.0) 
                obs, _ = env.reset()

            # 실시간 동기화를 위한 정밀한 시간 조절
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    run_test()