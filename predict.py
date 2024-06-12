import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# 저장된 모델 불러오기
model = load_model('cifar10_model.h5')

# CIFAR-10 클래스 레이블
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_preprocess_image(img_path):
    # 이미지 로드
    img = Image.open(img_path)
    # 이미지 크기를 CIFAR-10 크기(32x32)로 변경
    img = img.resize((32, 32))
    # 이미지를 numpy 배열로 변환
    img_array = np.array(img)
    # 이미지 전처리
    img_array = img_array / 255.0
    # 배치 차원 추가
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 이미지 경로
img_path = 'images.jpeg'

# 이미지 로드 및 전처리
new_image = load_and_preprocess_image(img_path)

# 이미지 시각화
plt.imshow(new_image[0])
plt.show()

# 이미지 예측
predictions = model.predict(new_image)
predicted_class = np.argmax(predictions, axis=1)

print("Predicted class:", class_names[predicted_class[0]])
