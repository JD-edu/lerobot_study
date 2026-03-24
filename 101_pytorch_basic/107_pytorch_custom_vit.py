import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
import os

# 1. 환경 설정 및 하이퍼파라미터
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
DATA_PATH = './images'  # 데이터 경로: train/val 폴더 밑에 can, cup, pet 폴더가 있어야 함

# 2. 데이터 전처리 및 증강 (Data Augmentation)
# 커스텀 데이터는 양이 적을 수 있으므로 RandomResizedCrop과 Flip을 추가합니다.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 표준값
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. 데이터 로드
# 기존 3단계(데이터 로드) 부분을 아래와 같이 수정하세요.

# 1. 전체 데이터 로드
full_dataset = datasets.ImageFolder(DATA_PATH, transform=data_transforms['train'])

# 2. 학습/검증 데이터 분할 (예: 80% 학습, 20% 검증)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# 3. 검증 데이터에는 전처리를 'val'용으로 교체 (중요!)
# random_split은 인덱스만 나누므로, 실제 transform을 바꾸려면 아래 작업이 필요합니다.
class CustomSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

# 최종 데이터셋 정의
# 주의: full_dataset 선언 시 transform을 None으로 하거나, 
# 아래처럼 CustomSubset을 통해 각각 적용하는 것이 정석입니다.
full_dataset.transform = None # 원본의 transform 초기화
train_dataset = CustomSubset(train_dataset, transform=data_transforms['train'])
val_dataset = CustomSubset(val_dataset, transform=data_transforms['val'])

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes
print(f"Classes: {class_names}")

# 4. 모델 설정 (ViT-Tiny 사용)
# num_classes를 3으로 설정하여 출력 레이어를 자동으로 변경합니다.
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=len(class_names))
model = model.to(device)

# 5. 손실 함수 및 최적화
criterion = nn.CrossEntropyLoss()
# ViT는 AdamW와 시너지가 좋습니다.
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 6. 학습 루프
print("Starting Training...")

# 6. 학습 루프 (시각화 리스트 추가)
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("Starting Training...")
for epoch in range(EPOCHS):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        # 결과 저장
        if phase == 'train':
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.item())
        else:
            history['val_loss'].append(epoch_loss)
            history['val_acc'].append(epoch_acc.item())

        print(f'Epoch {epoch+1}/{EPOCHS} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# 6-1. 결과 시각화 (그래프 출력)
plt.figure(figsize=(12, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 8. 개별 이미지 추론 (Inference) 테스트
def predict_image(image_path):
    from PIL import Image
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img_t = data_transforms['val'](img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_t)
        _, predicted = torch.max(output, 1)
        print(f'Prediction: {class_names[predicted[0]]}')

# 사용 예시: predict_image('test_pet.jpg')
torch.save(model.state_dict(), "vit_custom.pth")