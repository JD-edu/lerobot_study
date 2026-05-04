import time
from motor_control import MiniFeetechDriver


if __name__ == "__main__":
    PORT = "/dev/ttyUSB0"
    BAUDRATE = 1000000
    MOTOR_IDS = [1, 2, 3, 4, 5, 6]

    driver = MiniFeetechDriver(PORT, BAUDRATE)

    DEG_TO_TICK = 4096.0 / 360.0
    SWING_DEG = 5
    SWING_TICK = int(SWING_DEG * DEG_TO_TICK)

    STEP_TICK = 2
    STEP_DELAY = 0.03
    HOLD_TIME = 0.3

    try:
        print("[1] 현재 위치를 기준 위치로 읽습니다.")

        center_positions = {}

        for motor_id in MOTOR_IDS:
            pos = driver.get_position_filtered(motor_id, samples=5)

            if pos is None:
                print(f"ID {motor_id}: 위치 읽기 실패")
                driver.close()
                exit()

            center_positions[motor_id] = pos
            driver.set_torque(motor_id, True)
            time.sleep(0.05)

        print("기준 위치:", center_positions)
        print("[2] 각 관절을 현재 위치 기준 ±5도 천천히 움직입니다.")
        print("Ctrl+C로 종료")

        while True:
            # +5도 방향
            for step in range(0, SWING_TICK + 1, STEP_TICK):
                for motor_id in MOTOR_IDS:
                    target = center_positions[motor_id] + step
                    target = max(0, min(4095, target))
                    driver.set_position(motor_id, target)

                time.sleep(STEP_DELAY)

            time.sleep(HOLD_TIME)

            # -5도 방향
            for step in range(SWING_TICK, -SWING_TICK - 1, -STEP_TICK):
                for motor_id in MOTOR_IDS:
                    target = center_positions[motor_id] + step
                    target = max(0, min(4095, target))
                    driver.set_position(motor_id, target)

                time.sleep(STEP_DELAY)

            time.sleep(HOLD_TIME)

            # 다시 중심
            for step in range(-SWING_TICK, 1, STEP_TICK):
                for motor_id in MOTOR_IDS:
                    target = center_positions[motor_id] + step
                    target = max(0, min(4095, target))
                    driver.set_position(motor_id, target)

                time.sleep(STEP_DELAY)

            time.sleep(HOLD_TIME)

    except KeyboardInterrupt:
        print("\n종료합니다.")

    finally:
        driver.close()