
import time
from motor_control import MiniFeetechDriver

def soft_sweep_test():
    # --- 설정 영역 ---
    SERIAL_PORT = '/dev/ttyUSB0'
    BAUDRATE = 1000000
    MOTOR_IDS = [1, 2, 3, 4, 5, 6]  # 테스트할 모터 ID 리스트
    CENTER = 2048                   # 원점
    DEGREE_10 = 114                 # 10도에 해당하는 Step 값
    
    driver = MiniFeetechDriver(port=SERIAL_PORT, baudrate=BAUDRATE)

    driver.load_all_offsets(MOTOR_IDS)

    print("="*60)
    print("JD-101 Joint Sweep Test: +/- 10 Degrees")
    print("="*60)

    # 1. 초기화: 모든 모터를 센터로 이동
    print("\n[단계 1] 오프셋 제어를 통해 원점으로 정렬합니다.")
    for m_id in MOTOR_IDS:
        driver.set_torque(m_id, True)
        # 이제 2048은 우리가 물리적으로 맞춘 그 위치가 됩니다.
        driver.set_offset_position(m_id, CENTER) 
    time.sleep(2)

    # 2. 관절별 순차 테스트
    '''try:
        for m_id in MOTOR_IDS:
            print(f"\n>> ID {m_id:02d} 관절 테스트 시작...")
            
            # +10도 방향으로 천천히 이동 (1도씩 10단계)
            print(f"   ID {m_id:02d}: 0 -> +10도")
            for step in range(0, DEGREE_10 + 1, 10): 
                driver.set_position(m_id, CENTER + step)
                time.sleep(0.05)
            
            time.sleep(0.5)

            # +10도 -> -10도 방향으로 이동
            print(f"   ID {m_id:02d}: +10 -> -10도")
            for step in range(DEGREE_10, -DEGREE_10 - 1, -10):
                driver.set_position(m_id, CENTER + step)
                time.sleep(0.05)

            time.sleep(0.5)

            # -10도 -> 0도(센터)로 복귀
            print(f"   ID {m_id:02d}: -10 -> 0도 복귀")
            for step in range(-DEGREE_10, 0 + 1, 10):
                driver.set_position(m_id, CENTER + step)
                time.sleep(0.05)
            
            print(f"ID {m_id:02d} 테스트 완료.")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다. 토크를 해제합니다.")
        for m_id in MOTOR_IDS:
            driver.set_torque(m_id, False)'''

    print("\n" + "="*60)
    print("모든 관절 테스트가 완료되었습니다!")
    print("="*60)

if __name__ == "__main__":
    soft_sweep_test()