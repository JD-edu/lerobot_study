import time
import json
from pathlib import Path
from motor_control import MiniFeetechDriver


MOTOR_TABLE = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


def calibrate_single_motor(
    port="/dev/ttyUSB0",
    joint_name="shoulder_pan",
    out_json_path="./single_motor_calibration.json",
):
    driver = MiniFeetechDriver(port=port)

    if joint_name not in MOTOR_TABLE:
        raise ValueError(f"Unknown joint_name={joint_name}. Choose one of {list(MOTOR_TABLE.keys())}")

    motor_id = MOTOR_TABLE[joint_name]

    # STS3215가 0~4095라고 가정(12bit)
    MAX_RES = 4095
    HALF_TURN = MAX_RES // 2  # 2047

    print(f"\n[선택됨] joint={joint_name}, motor_id={motor_id}")
    print("\n[0] 토크 OFF (손으로 관절을 움직이기 위해)")
    driver.set_torque(motor_id, False)
    time.sleep(0.1)

    # ---- Step 1: homing offset ----
    input(f"\n[1] {joint_name}(모터 {motor_id})을 가능한 '중앙' 위치에 두고 ENTER")
    pos_center = driver.get_position(motor_id)
    if pos_center is None:
        raise RuntimeError("현재 위치를 읽지 못했습니다.")

    homing_offset = int(pos_center) - int(HALF_TURN)

    print(f"  - 현재 pos = {pos_center}")
    print(f"  - HALF_TURN = {HALF_TURN}")
    print(f"  - 계산된 homing_offset = {homing_offset}")

    print("\n  -> 모터에 Homing_Offset 기록")
    driver.set_homing_offset(motor_id, homing_offset)
    time.sleep(0.05)

    # ---- Step 2: range min/max ----
    input(f"\n[2] {joint_name}을 '한쪽 끝'까지 천천히 돌리고 ENTER")
    pos_a = driver.get_position(motor_id)
    if pos_a is None:
        raise RuntimeError("끝 위치(A)를 읽지 못했습니다.")

    input(f"\n[3] {joint_name}을 '반대쪽 끝'까지 천천히 돌리고 ENTER")
    pos_b = driver.get_position(motor_id)
    if pos_b is None:
        raise RuntimeError("끝 위치(B)를 읽지 못했습니다.")

    range_min = int(min(pos_a, pos_b))
    range_max = int(max(pos_a, pos_b))

    print(f"\n  - 기록된 raw min = {range_min}")
    print(f"  - 기록된 raw max = {range_max}")

    if range_min == range_max:
        raise ValueError("min과 max가 같습니다. 관절을 충분히 움직였는지 확인하세요.")

    print("\n  -> 모터에 Min/Max_Position_Limit 기록")
    driver.set_position_limits(motor_id, range_min, range_max)

    # ---- Save JSON ----
    calib = {
        joint_name: {
            "id": motor_id,
            "drive_mode": 0,
            "homing_offset": homing_offset,
            "range_min": range_min,
            "range_max": range_max,
        }
    }

    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=4, ensure_ascii=False)

    print(f"\n[완료] 캘리브레이션 JSON 저장: {out_json_path}")
    return calib


def choose_joint_interactively():
    joints = list(MOTOR_TABLE.keys())
    print("\n칼리브레이션할 모터를 선택하세요:")
    for i, name in enumerate(joints, start=1):
        print(f"  {i}) {name} (id={MOTOR_TABLE[name]})")

    s = input("\n번호 입력 (1~6): ").strip()
    if not s.isdigit():
        raise ValueError("숫자를 입력해야 합니다.")
    idx = int(s)
    if not (1 <= idx <= len(joints)):
        raise ValueError("범위를 벗어났습니다.")
    return joints[idx - 1]


if __name__ == "__main__":
    joint = choose_joint_interactively()
    calibrate_single_motor(
        port="/dev/ttyUSB0",
        joint_name=joint,
        out_json_path=f"./{joint}_calibration_leader.json",
    )
