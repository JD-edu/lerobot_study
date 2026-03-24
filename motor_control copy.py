import serial
import time
import struct

class MiniFeetechDriver:
    # Feetech 모터 주요 레지스터 주소 (STS3215 기준)
    REG_GOAL_POSITION = 42
    REG_PRESENT_POSITION = 56
    REG_TORQUE_ENABLE = 40

    # ===== 캘리브레이션용(추가) =====
    # 아래 3개 주소는 STS3215 컨트롤테이블에 따라 달라질 수 있습니다.
    # 필요시 본인 모터 문서/테이블로 확인해서 수정하세요.
    REG_HOMING_OFFSET = 31
    REG_MIN_POSITION_LIMIT = 9
    REG_MAX_POSITION_LIMIT = 11

    def __init__(self, port='/dev/ttyUSB0', baudrate=1000000):
        # 1. 시리얼 포트 설정 (LeRobot도 1M baudrate 사용)
        self.ser = serial.Serial(port, baudrate, timeout=0.05)

     
    def _send_packet(self, motor_id, instruction, parameters):
        """Feetech 프로토콜에 맞춘 패킷 생성 및 전송"""
        length = len(parameters) + 2
        packet = [0xFF, 0xFF, motor_id, length, instruction] + parameters
        
        # Checksum 계산 (LeRobot의 _checksum 함수와 동일한 논리)
        checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
        packet.append(checksum)
        
        self.ser.write(bytearray(packet))
        return self.ser.read(100) # 응답 패킷 읽기 (생략 가능하나 확인용으로 권장)
    
    
    # ---- Generic read/write helpers (추가) ----
    def write_u8(self, motor_id, reg_addr, value):
        self._send_packet(motor_id, 0x03, [reg_addr & 0xFF, value & 0xFF])

    def write_u16(self, motor_id, reg_addr, value):
        lo = value & 0xFF
        hi = (value >> 8) & 0xFF
        self._send_packet(motor_id, 0x03, [reg_addr & 0xFF, lo, hi])

    def read_u16(self, motor_id, reg_addr):
        resp = self._send_packet(motor_id, 0x02, [reg_addr & 0xFF, 2])
        # FF FF ID LEN ERR VAL_L VAL_H CHK (최소 8바이트)
        if len(resp) >= 8:
            lo = resp[5]
            hi = resp[6]
            return (hi << 8) | lo
        return None

    def set_torque(self, motor_id, enable):
        """모터의 토크를 켜거나 끕니다 (1: On, 0: Off)"""
        # [주소, 값] 순서로 파라미터 전달
        self._send_packet(motor_id, 0x03, [self.REG_TORQUE_ENABLE, 1 if enable else 0])

    def set_position(self, motor_id, position):
        """목표 위치로 이동 (position: 0 ~ 4095)"""
        # 16비트 데이터를 2개의 8비트로 분리 (Little Endian)
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        self._send_packet(motor_id, 0x03, [self.REG_GOAL_POSITION, pos_low, pos_high])

    def get_position(self, motor_id):
        """현재 위치 값을 읽어옴"""
        # 0x02: READ instruction (주소 56번부터 2바이트 읽기)
        response = self._send_packet(motor_id, 0x02, [self.REG_PRESENT_POSITION, 2])
        
        if len(response) >= 8: # 패킷 구조: FF FF ID LEN ERR VAL_L VAL_H CHK
            # 패킷 구조: FF FF ID LEN ERR VAL_L VAL_H CHK
            pos_low = response[5]
            pos_high = response[6]
            
            # 1. 비트 연산으로 합치기
            raw_pos = (pos_high << 8) | pos_low
            
            # 2. 12비트 마스킹 (0x0FFF = 4095)
            # 4095를 넘는 모든 상위 비트를 0으로 만듭니다.
            clean_pos = raw_pos & 0x0FFF
            
            return clean_pos
        return None
    
     # ---- 캘리브레이션 레지스터 write/read (추가) ----
    def set_homing_offset(self, motor_id, offset):
        # Feetech의 homing offset은 부호가 있을 수 있습니다(모터/프로토콜에 따라).
        # 교육용 단순화: 16bit로 기록. (필요하면 signed 처리 추가)
        self.write_u16(motor_id, self.REG_HOMING_OFFSET, int(offset) & 0xFFFF)

    def set_position_limits(self, motor_id, min_pos, max_pos):
        self.write_u16(motor_id, self.REG_MIN_POSITION_LIMIT, int(min_pos))
        self.write_u16(motor_id, self.REG_MAX_POSITION_LIMIT, int(max_pos))

    def read_position_limits(self, motor_id):
        mn = self.read_u16(motor_id, self.REG_MIN_POSITION_LIMIT)
        mx = self.read_u16(motor_id, self.REG_MAX_POSITION_LIMIT)
        return mn, mx

