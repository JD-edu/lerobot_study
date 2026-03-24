import cv2

# 0번 카메라 연결 (내장 웹캠 또는 첫 번째 USB 카메라)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        print("프레임을 가져올 수 없습니다. 종료합니다.")
        break

    # 화면에 출력
    cv2.imshow('Camera Test', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
