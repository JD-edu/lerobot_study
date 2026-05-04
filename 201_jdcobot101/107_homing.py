import time
from motor_control import MiniFeetechDriver


if __name__ == "__main__":
    PORT = "/dev/ttyUSB0"
    BAUDRATE = 1000000

    # 그리퍼 제외 5개 축
    MOTOR_IDS = [1, 2, 3, 4, 5]

    LOGICAL_CENTER = 2048
    STEP_TICK = 3
    STEP_DELAY = 0.03

    driver = MiniFeetechDriver(PORT, BAUDRATE)

    try:
        current_positions = {}
        offsets = {}
        targets = {}

        print("[1] 현재 위치와 offset 읽기")

        for motor_id in MOTOR_IDS:
            pos = driver.get_position_filtered(motor_id, samples=5)
            offset = driver.get_homing_offset(motor_id)

            if pos is None or offset is None:
                print(f"ID{motor_id}: 읽기 실패")
                driver.close()
                exit()

            # logical = physical + offset
            # physical = logical - offset
            target = (LOGICAL_CENTER - offset) % 4096

            current_positions[motor_id] = pos
            offsets[motor_id] = offset
            targets[motor_id] = target

            driver.set_torque(motor_id, True)

            print(
                f"ID{motor_id}: raw={pos:4d}, "
                f"offset={offset:6d}, "
                f"target={target:4d}"
            )

            time.sleep(0.05)

        print("\n[2] 5개 서보를 logical 2048 위치로 천천히 이동")

        moving = True

        while moving:
            moving = False
            line = []

            for motor_id in MOTOR_IDS:
                current = current_positions[motor_id]
                target = targets[motor_id]
                offset = offsets[motor_id]

                if current < target:
                    current += STEP_TICK
                    if current > target:
                        current = target
                    moving = True

                elif current > target:
                    current -= STEP_TICK
                    if current < target:
                        current = target
                    moving = True

                current_positions[motor_id] = current

                driver.set_position(motor_id, int(current))

                logical = (current + offset) % 4096

                line.append(
                    f"ID{motor_id}:R{current:4d}/L{logical:4d}"
                )

                time.sleep(0.005)

            print(" | ".join(line))
            time.sleep(STEP_DELAY)

        print("\n5개 서보가 logical 2048 위치로 이동 완료.")

    except KeyboardInterrupt:
        print("\n사용자 중단.")

    finally:
        driver.close()