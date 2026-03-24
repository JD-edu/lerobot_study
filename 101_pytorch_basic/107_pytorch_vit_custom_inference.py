import cv2
import torch
import timm
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

# 1. 모델 및 클래스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 시 폴더 순서(알파벳 순)와 동일하게 작성하세요.
# 예: ['can', 'cup', 'pet']
class_names = ['can', 'cup', 'pet'] 
num_classes = len(class_names)

def load_custom_vit(model_path):
    # 학습 시 사용한 모델 구조와 동일하게 생성 (num_classes 포함)
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    
    # 가중치 파일 로드
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"모델 '{model_path}' 로드 성공!")
    except FileNotFoundError:
        print(f"에러: '{model_path}' 파일을 찾을 수 없습니다.")
        exit()
        
    model = model.to(device)
    model.eval()
    return model

# 2. 전처리 설정 (학습 시 'val' transform과 동일하게 설정)
# ImageNet 표준 정규화 값이 포함되어야 합니다.
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 모델 로드
MODEL_FILE = "vit_custom.pth"
model = load_custom_vit(MODEL_FILE)

# 3. 실시간 웹캠 추론
cap = cv2.VideoCapture(0)

print("ViT 실시간 추론 시작... 종료하려면 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 화면 중앙 ROI 설정 (ViT는 정중앙을 잘 잡는 것이 중요합니다)
    h, w, _ = frame.shape
    roi_size = 400
    x1, y1 = (w - roi_size) // 2, (h - roi_size) // 2
    x2, y2 = x1 + roi_size, y1 + roi_size
    roi = frame[y1:y2, x1:x2]

    # 추론 준비 (BGR -> RGB 변환)
    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # 모델 예측
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        
        label = class_names[pred.item()]
        score = conf.item()

    # 결과 시각화
    # 확신도 85% 이상일 때만 초록색, 아니면 주황색 표시
    color = (0, 255, 0) if score > 0.85 else (0, 165, 255)
    display_text = f"ViT: {label} ({score*100:.1f}%)"
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, display_text, (x1, y1-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Custom ViT Live Inference", frame)
    cv2.imshow("Input to Model", roi) # 모델이 집중하는 영역 확인용

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()