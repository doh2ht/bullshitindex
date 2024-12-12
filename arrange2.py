from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기 및 설정 (리사이즈 없이 원본 이미지 사용)
image_path = "/Users/kimdohee/Documents/2024/newlyformed/5.arrange/page/meme/06.jpg"
img = Image.open(image_path)

# 원본 이미지의 너비와 높이 얻기
width, height = img.size

# 이미지 데이터를 numpy 배열로 변환 (정수형으로 변환하여 색상 손실 방지)
img_array = np.array(img).astype('float32')

# 그리드 크기 및 변수 설정
grid_width = 1   # 가로 그리드 크기 설정 (픽셀 단위)
grid_height = 1000  # 세로 그리드 크기 설정 (픽셀 단위)

# 결과 이미지 생성 (float32 타입으로 초기화)
gradient_image = np.zeros_like(img_array, dtype='float32')

# 그리드별 평균 색상 계산 및 적용
for col in range(0, width, grid_width):
    for row in range(0, height, grid_height):
        # 각 그리드의 평균 색상 계산
        left_color = img_array[row:row + grid_height, col].mean(axis=0)  # 그리드 내 평균 색상 계산
        
        # 좌측 색상만 그대로 적용 (보간 계산 생략)
        gradient_image[row:row + grid_height, col] = left_color

# float32 타입의 결과 이미지를 uint8로 변환하기 전에 값 범위 조정
gradient_image = np.clip(np.round(gradient_image), 0, 255).astype('uint8')

# numpy 배열을 PIL 이미지로 변환
gradient_img = Image.fromarray(gradient_image)

# 결과 이미지 출력: figsize는 이미지 크기와 동일하게 설정
fig, ax = plt.subplots(1, 2, figsize=(width / 100, height / 100), dpi=100)
ax[0].imshow(img)
ax[0].axis('off')

ax[1].imshow(gradient_img)
ax[1].axis('off')

# 여백을 최소화하여 이미지의 실제 크기로 표시
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.gca().set_position([0, 0, 1, 1])  # 그래프를 전체 창에 맞게 조정
plt.margins(0, 0)  # 마진 제거
plt.show()
