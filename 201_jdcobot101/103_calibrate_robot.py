import time
from motor_control import MiniFeetechDriver 

def run_homing_sequence():
    # --- 설정 영역 ---
    SERIAL_PORT = '/dev/ttyUSB0'
    BAUDRATE = 1000000
    MOTOR_IDS = [1, 2, 3, 4, 5]  # 5축 로봇 ID
    
    driver = MiniFeetechDriver(port=SERIAL_PORT, baudrate=BAUDRATE)
    
    print("="*50)
    print("JD-101 (5-Axis) Homing & Calibration")
    print("="*50)
    
    # 1. 토크 해제 (Manual Positioning 준비)
    print("\n[단계 1] 토크를 해제합니다. 로봇을 손으로 붙잡으세요.")
    input("준비가 되었다면 [Enter]를 누르세요...")
    
    for m_id in MOTOR_IDS:
        driver.set_torque(m_id, False)
    print(">> 모든 모터의 토크가 해제되었습니다.")

    # 2. 수동 정렬 대기
    print("\n[단계 2] 로봇을 수직(또는 원하는 원점 자세)으로 정렬하세요.")
    print("축 정렬이 완료되면 현재 위치를 '0점'으로 기록합니다.")
    input("정렬을 마쳤다면 [Enter]를 누르세요...")

    # 3. 현재 위치 읽기 및 저장
    home_offsets = {}
    print("\n[단계 3] 현재 위치 읽는 중...")
    for m_id in MOTOR_IDS:
        pos = driver.get_position(m_id)
        if pos is not None:
            home_offsets[m_id] = pos
            print(f"ID {m_id:02d}의 현재 원점 원시값(Raw): {pos}")
        else:
            print(f"ID {m_id:02d} 읽기 실패! 연결을 확인하세요.")

    # 4. 토크 인가 (현재 위치에서 고정)
    print("\n[단계 4] 현재 위치에서 모터를 고정(Freeze)합니다.")
    for m_id in MOTOR_IDS:
        if m_id in home_offsets:
            # 현재 위치를 목표 위치로 먼저 설정 후 토크 온
            driver.set_position(m_id, home_offsets[m_id])
            driver.set_torque(m_id, True)
    
    print(">> 모든 모터가 고정되었습니다.")

    # 5. 결과 요약 및 가이드
    print("\n" + "="*50)
    print("홈잉 작업이 완료되었습니다!")
    print("="*50)
    print("아래의 값을 소스코드의 OFFSET 상수로 저장하여 사용하세요:")
    print(f"HOME_RAW_VALUES = {home_offsets}")
    print("-" * 50)
    
    # 추가 팁: 하드웨어 레지스터에 기록하고 싶은 경우
    save_hw = input("이 값을 모터 내부 레지스터(Homing Offset)에 영구 저장할까요? (y/n): ")
    if save_hw.lower() == 'y':
        for m_id, pos in home_offsets.items():
            # 1. 오프셋 계산
            diff = 2048 - pos
            
            # 2. 하드웨어 레지스터에 오프셋 저장
            driver.set_homing_offset(m_id, diff)
            
            # 3. [중요] 오프셋이 적용된 직후, 목표 위치를 2048(새로운 중심)로 강제 설정
            # 이렇게 해야 모터가 '좌표가 바뀌었네? 그럼 바뀐 좌표의 제자리(2048)에 있자'라고 판단합니다.
            driver.set_position(m_id, 2048)
            
            print(f"ID {m_id:02d}: Offset {diff} 저장 및 위치 동기화 완료")
        
        print(">> 하드웨어 오프셋 설정 완료. 이제 모든 축의 수직 상태가 2048입니다.")

if __name__ == "__main__":
    try:
        run_homing_sequence()
    except Exception as e:
        print(f"\n오류 발생: {e}")
    except KeyboardInterrupt:
        print("\n사용자에 의해 취소되었습니다.")