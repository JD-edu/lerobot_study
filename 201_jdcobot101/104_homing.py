import time
from motor_control import MiniFeetechDriver 

def calibrate_and_save_homing():
    # --- 설정 영역 ---
    SERIAL_PORT = '/dev/ttyUSB0'
    BAUDRATE = 1000000
    # 그리퍼를 포함한 6축 로봇 ID (예: 1~6)
    MOTOR_IDS = [1, 2, 3, 4, 5, 6] 
    CENTER_VALUE = 2048 # STS3215의 논리적 중앙값 (0~4095 기준)

    driver = MiniFeetechDriver(port=SERIAL_PORT, baudrate=BAUDRATE)
    
    print("="*60)
    print("JD-101 6-Axis Robot: Hardware Homing Calibration")
    print("="*60)

    # 1. 수동 정렬을 위한 토크 해제
    print("\n[1단계] 토크를 해제합니다. 로봇을 원점 자세로 정렬하세요.")
    for m_id in MOTOR_IDS:
        driver.set_torque(m_id, False)
    
    input(">> 정렬이 완료되었다면 [Enter]를 누르세요. (현재 위치가 2048이 됩니다)")

    # 2. 현재 위치 읽기 및 오프셋 계산/저장
    print("\n[2단계] 하드웨어 레지스터에 오프셋 기록 중...")
    
    for m_id in MOTOR_IDS:
        current_pos = driver.get_position(m_id)
        
        if current_pos is not None:
            # 오프셋 계산: (목표값 - 현재값)
            # STS3215 내부적으로 현재값에 이 오프셋을 더해 출력을 결정합니다.
            offset = CENTER_VALUE - current_pos
            
            try:
                # EPROM 쓰기 잠금 해제 (필요한 경우)
                driver.unLockEprom(m_id)
                
                # Homing Offset 레지스터에 기록
                # 주의: 라이브러리에 따라 set_homing_offset 함수의 인자 순서가 다를 수 있습니다.
                driver.set_homing_offset(m_id, offset)
                
                # 쓰기 잠금 다시 설정
                driver.lockEprom(m_id)
                
                print(f"ID {m_id:02d}: 현재값({current_pos}) -> 오프셋({offset}) 저장 완료")
            except Exception as e:
                print(f"ID {m_id:02d}: 저장 실패 - {e}")
        else:
            print(f"ID {m_id:02d}: 위치 읽기 실패! 통신 상태를 확인하세요.")

    # 3. 오프셋 적용 확인 (동기화)
    print("\n[3단계] 오프셋 적용 확인 및 고정(Freeze)")
    for m_id in MOTOR_IDS:
        # 오프셋이 적용된 후이므로, 2048로 이동 명령을 내리면 그 자리에 고정됩니다.
        driver.set_position(m_id, CENTER_VALUE+5)
        driver.set_torque(m_id, True)

    print("\n" + "="*60)
    print("모든 축의 하드웨어 원점 설정이 완료되었습니다.")
    print("이제 전원을 재시작해도 현재 자세가 2048(중앙)로 인식됩니다.")
    print("="*60)

if __name__ == "__main__":
    calibrate_and_save_homing()