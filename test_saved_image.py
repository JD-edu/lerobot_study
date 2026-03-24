import cv2
import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

# 1. 데이터셋 로드
repo_id = "my_robot_task"
local_root = Path(Path.home(), ".cache/huggingface/lerobot", repo_id)

if local_root.exists():
    # 로컬 경로를 직접 지정하여 불러오기
    dataset = LeRobotDataset(repo_id, root=local_root)
else:
    print(f"❌ 로컬 경로를 찾을 수 없습니다: {local_root}")
    # 경로가 다를 경우 직접 지정 가능: root="/home/robo/data/my_robot_task"

print(f"=== 데이터셋 요약: {repo_id} ===")
print(f"총 프레임 수: {len(dataset)}")
print(f"총 에피소드 수: {dataset.num_episodes}")
print(f"평균 FPS: {dataset.fps}")
print("-" * 50)

# 2. 에피소드 인덱스 정보 가져오기 (최신 API 방식)
# hf_dataset의 'episode_index' 열을 이용해 각 에피소드의 시작/끝 인덱스를 계산합니다.
episode_idx = 0  # 확인하고 싶은 에피소드 번호 (0~7)

# 해당 에피소드에 해당하는 프레임들만 필터링하여 인덱스 추출
all_episode_indices = np.array(dataset.hf_dataset["episode_index"])
frame_indices = np.where(all_episode_indices == episode_idx)[0]

if len(frame_indices) == 0:
    print(f"에피소드 {episode_idx}를 찾을 수 없습니다.")
else:
    from_idx = int(frame_indices[0])
    to_idx = int(frame_indices[-1]) + 1
    episode_len = to_idx - from_idx

    print(f"[에피소드 {episode_idx} 상세 분석]")
    print(f"프레임 범위: {from_idx} ~ {to_idx - 1} (총 {episode_len} 프레임)")

    # 3. 데이터 Shape 및 샘플 확인
    sample_frame = dataset[from_idx]
    print(f"\n[텐서 Shape 확인]")
    print(f"- 이미지: {sample_frame['observation.image'].shape} (C, H, W)")
    print(f"- 상태(State): {sample_frame['observation.state'].shape}")
    print(f"- 액션(Action): {sample_frame['action'].shape}")

    # 4. 이미지 시퀀스 재생
    print(f"\n>> 에피소드 {episode_idx} 재생 중... (창에서 'q'를 누르면 종료)")

    for i in range(from_idx, to_idx):
        frame_data = dataset[i]
        
        # 이미지 변환 (Tensor -> BGR)
        img = frame_data["observation.image"].permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 텍스트 오버레이
        state_val = frame_data["observation.state"]
        cv2.putText(img, f"Ep: {episode_idx} | Frame: {i}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Dataset Review", img)
        
        if i % 30 == 0: # 1초(30프레임)마다 터미널에 상태 출력
            print(f"Frame {i} State: {state_val.tolist()}")

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()