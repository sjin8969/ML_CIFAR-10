import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

model = load_model('cifar10_model.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
# 이미지 경로는 본인의 환경에 맞게 설정을 해야함
img_path = 'images.jpeg'

new_image = load_and_preprocess_image(img_path)

plt.imshow(new_image[0])
plt.show()

predictions = model.predict(new_image)
predicted_class = np.argmax(predictions, axis=1)

print("Predicted class:", class_names[predicted_class[0]])
