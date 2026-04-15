import gymnasium as gym
import numpy as np
import torch
import mujoco as mj
import mujoco.viewer
from stable_baselines3 import PPO
import time
from gymnasium import spaces
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import torch.nn as nn

# [주의] 이전에 정의하신 ACTPolicy 클래스가 파일 내에 정의되어 있어야 합니다.
class ACTPolicy(nn.Module):
    def __init__(self, action_dim=6, state_dim=6, chunk_size=50, hidden_dim=512, nheads=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # A. 시각 백본 (ResNet18)
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # B. State Encoder (현재 상태 6차원 -> 512차원 확장)
        self.state_encoder = nn.Linear(state_dim, hidden_dim)

        # C. CVAE Encoder (이미지 특징 512 + 액션 뭉치 300 = 812 입력)
        self.cvae_encoder = nn.Linear(hidden_dim + (action_dim * chunk_size), hidden_dim * 2)
        
        # D. Transformer (핵심 추론 엔진)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=nheads, 
            num_encoder_layers=4, num_decoder_layers=4, batch_first=True
        )

        # E. Action Head (미래 궤적 출력)
        self.action_head = nn.Linear(hidden_dim, action_dim * chunk_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, state, actions=None):
        batch_size = image.shape[0]
        image = self.normalize(image)
        img_feat = self.backbone(image) # (B, 512)
        
        # State를 512차원으로 투영
        state_feat = self.state_encoder(state) # (B, 512)

        latent_loss = torch.tensor(0.0).to(image.device)
        if actions is not None:
            # [차원 맞춤] 만약 데이터로더에서 1개 액션만 왔다면 50개로 확장
            if actions.ndim == 2: # (B, 6)
                actions = actions.unsqueeze(1).repeat(1, self.chunk_size, 1)
            
            flattened_actions = actions.reshape(batch_size, -1) # (B, 300)
            combined = torch.cat([img_feat, flattened_actions], dim=1) # (B, 812)
            
            h = F.relu(self.cvae_encoder(combined))
            mu, logvar = torch.chunk(h, 2, dim=1)
            z = self.reparameterize(mu, logvar) # (B, 512)
            latent_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            z = torch.zeros(batch_size, 512).to(image.device)

        # Transformer 입력 준비: [Style(z), Image, State] 모두 (B, 512)
        # stack 결과: (B, 3, 512)
        src = torch.stack([z, img_feat, state_feat], dim=1) 
        
        # Transformer 통과
        out = self.transformer.encoder(src)
        # Style 토큰(0번 인덱스) 결과물을 사용하여 액션 예측
        pred_actions = self.action_head(out[:, 0, :]) 
        
        return pred_actions.view(batch_size, self.chunk_size, self.action_dim), latent_loss


# [필수] 앞서 정의한 ACTPolicy와 SO101HybridEnv 클래스가 동일한 파일에 있거나 import 되어야 합니다.
class SO101HybridEnv(gym.Env):
    def __init__(self, checkpoint_path="act_robot_model.pth", xml_path="scene.xml"):
        super().__init__()
        
        # 1. MuJoCo 로드
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. 사전 학습된 ACT 모델 로드 (Base Policy)
        self.act_policy = ACTPolicy().to(self.device)
        self.act_policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.act_policy.eval() # 추론 모드
        
        # 3. Action Space: RL은 ACT 출력값에 더해질 '미세 보정값'만 결정 (라디안 단위)
        # 예: 각 관절당 ±0.05 rad (약 3도) 이내에서만 보정 허용
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(6,), dtype=np.float32)
        
        # 4. Observation Space: 현재 관절(6) + 속도(6) + 목표물 위치(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def _get_obs(self):
        # 큐브 위치 가져오기
        cube_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "red_cube")
        cube_pos = self.data.geom_xpos[cube_id]
        return np.concatenate([self.data.qpos[:6], self.data.qvel[:6], cube_pos]).astype(np.float32)

    def _get_act_base(self):
        """ACT 모델로부터 현재 상태에 대한 기본 궤적 추출"""
        curr_qpos = torch.tensor(self.data.qpos[:6]).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dummy_img = torch.zeros((1, 3, 224, 224)).to(self.device)
            # 수정 부분: 튜플로 반환되는 값을 언패킹(Unpacking)합니다.
            pred_actions_all, _ = self.act_policy(dummy_img, curr_qpos)
            
            # pred_actions_all은 (batch, chunk_size, action_dim) 모양입니다.
            # 이 중 첫 번째 배치, 첫 번째 시점의 액션을 가져옵니다.
            return pred_actions_all[0, 0, :].cpu().numpy()

    def step(self, residual_action):
        # 1. ACT의 기본 예측값 가져오기
        base_action = self._get_act_base()
        
        # 2. 하이브리드 제어: ACT(큰 틀) + RL(미세 보정)
        final_target_qpos = base_action + residual_action
        
        # 3. MuJoCo 제어 주입 (PD Control)
        for i in range(6):
            self.data.ctrl[i] = 100.0 * (final_target_qpos[i] - self.data.qpos[i]) - 2.0 * self.data.qvel[i]
            
        mj.mj_step(self.model, self.data)
        
        # 4. 보상 계산
        ee_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "effector")
        cube_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "red_cube")
        dist = np.linalg.norm(self.data.site_xpos[ee_id] - self.data.geom_xpos[cube_id])
        
        # - 거리 보상 (가까울수록 0에 수렴)
        reward = -dist 
        
        # - 정밀 터치 성공 보상
        terminated = False
        if dist < 0.01:
            reward += 100.0
            terminated = True
            
        # - 급격한 움직임 제한 (정규화된 보정값에 대한 페널티)
        reward -= 0.1 * np.sum(np.square(residual_action))
        
        return self._get_obs(), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)
        
        # 큐브 위치를 가동 범위 내에서 랜덤하게 리셋하여 학습 효율 증대
        cube_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "red_cube")
        self.model.body_pos[cube_id] = [
            np.random.uniform(0.2, 0.4),
            np.random.uniform(-0.1, 0.1),
            0.05
        ]
        return self._get_obs(), {}

def run_hybrid_inference(model_path="so101_hybrid_fine_tuned.zip"):
    # 1. 하이브리드 환경 생성 (렌더링 모드 설정)
    # 학습 시와 동일한 체크포인트와 XML 경로를 사용합니다.
    env = SO101HybridEnv(checkpoint_path="act_robot_model.pth", xml_path="scene.xml")
    
    # 2. 학습된 PPO 모델 로드
    try:
        model = PPO.load(model_path, env=env, device="cpu")
        print(f"✅ 모델 로드 완료: {model_path}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 3. 시뮬레이션 실행 및 시각화
    obs, _ = env.reset()
    
    # launch_passive를 사용하여 뷰어를 실행합니다.
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("🚀 하이브리드 추론 시뮬레이션 시작...")
        
        while viewer.is_running():
            step_start = time.time()

            # A. RL 에이전트로부터 미세 보정값(Residual Action) 예측
            # deterministic=True로 설정하여 가장 확률 높은 최적의 보정값만 가져옵니다.
            residual_action, _states = model.predict(obs, deterministic=True)

            # B. 환경에 보정값 전달 (내부적으로 ACT의 출력과 합쳐짐)
            obs, reward, terminated, truncated, info = env.step(residual_action)

            # C. 화면 동기화
            viewer.sync()

            # D. 목표 도달 시 시각적 확인을 위해 잠시 멈춘 후 리셋
            if terminated or truncated:
                # 큐브에 닿았을 때의 거리 확인 (정확도 체크)
                ee_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_SITE, "effector")
                cube_id = mj.mj_name2id(env.model, mj.mjtObj.mjOBJ_GEOM, "red_cube")
                final_dist = np.linalg.norm(env.data.site_xpos[ee_id] - env.data.geom_xpos[cube_id])
                
                print(f"🎯 목표 도달! 최종 오차: {final_dist*1000:.2f} mm")
                time.sleep(1.5)
                obs, _ = env.reset()

            # 실시간 시뮬레이션 속도 유지를 위한 타이밍 조절
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    run_hybrid_inference()