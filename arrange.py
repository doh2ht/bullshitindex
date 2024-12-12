from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from skimage.color import rgb2lab, lab2rgb

# 이미지 불러오기 및 설정
image_path = "/Users/kimdohee/Documents/2024/newlyformed/5.arrange/page/meme/27.jpg"
img = Image.open(image_path)

# 원본 이미지 크기 계산
orig_width, orig_height = img.size
print(f"Original Image Size: {orig_width} x {orig_height}")

# 원하는 이미지 크기로 리사이즈 (비율 유지)
new_height = 560  # 원하는 높이 설정
new_width = int(new_height * (orig_width / orig_height))  # 비율에 따라 너비 계산
img = img.resize((new_width, new_height))

# 이미지 데이터를 numpy 배열로 변환
img_array = np.array(img)

# 그리드 크기 및 변수 설정
grid_width = 180   # 가로 그리드 크기 설정 (픽셀 단위)
grid_height = 180  # 세로 그리드 크기 설정 (픽셀 단위)
height, width, _ = img_array.shape

# 색상 평균값을 저장할 리스트 생성
avg_colors = []

# 그리드별 평균 색상 계산
for col in range(0, width, grid_width):
    for row in range(0, height, grid_height):
        # 각 그리드의 평균 색상 계산
        grid = img_array[row:row + grid_height, col:col + grid_width]
        avg_color = grid.mean(axis=(0, 1))  # 각 그리드의 평균 색상 (RGB)
        avg_colors.append(avg_color)

# RGB 색상을 Lab 색 공간으로 변환
avg_colors_lab = rgb2lab(np.array(avg_colors) / 255)  # 색상 값을 0-1 사이로 정규화 후 변환

# Lab 색상 간 거리 행렬 계산
distance_matrix = pairwise_distances(avg_colors_lab, metric='euclidean')

# 계층적 클러스터링을 사용하여 색상 간 유사도 기반 정렬
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=20).fit(avg_colors_lab)
labels = clustering.labels_

# 클러스터 내에서 유사한 색상끼리 정렬
sorted_indices = np.argsort(labels)
avg_colors_sorted = np.array(avg_colors_lab)[sorted_indices]  # Lab 색상으로 정렬

# 원본 이미지 크기에 맞는 figsize 설정
figsize_width = orig_width / 100  # figsize는 인치 단위, 원본 이미지 크기에 따라 조정
figsize_height = orig_height / 100

# 그라데이션 효과를 위한 보간 함수 정의 (Lab 색 공간에서 보간)
def interpolate_lab_colors(color1, color2, num_steps=100):
    """
    두 Lab 색상 사이의 그라데이션을 num_steps 단계로 생성.
    Args:
    - color1, color2: Lab 값으로 표현된 색상
    - num_steps: 보간할 단계 수
    Returns:
    - 보간된 Lab 색상 배열
    """
    return [color1 + (color2 - color1) * t / num_steps for t in range(num_steps)]

# Lab 색상 막대 그라데이션 적용
fig, ax = plt.subplots(figsize=(figsize_width, figsize_height), dpi=100)  # dpi=100 설정

# 전체 막대에 그라데이션 효과 적용
num_colors = len(avg_colors_sorted)
gradient_image = np.zeros((1, (num_colors - 1) * 100, 3))  # 마지막 색상 공간 없이 배열 생성
for i in range(num_colors - 1):
    # 현재 색상과 다음 색상 간의 그라데이션 계산
    gradient_section_lab = interpolate_lab_colors(avg_colors_sorted[i], avg_colors_sorted[i + 1])
    # Lab 색상을 RGB 색상으로 변환
    gradient_section_rgb = lab2rgb(np.array(gradient_section_lab).reshape(-1, 1, 3)).reshape(-1, 3)
    # 각 그라데이션 섹션을 이미지에 채우기
    gradient_image[0, i * 100: (i + 1) * 100] = gradient_section_rgb

# 최종 그라데이션을 PIL 이미지로 변환
gradient_image = np.uint8(gradient_image * 255)  # uint8 형식으로 변환 (0-255)
gradient_img = Image.fromarray(gradient_image)

# 그래프에 이미지 표시
ax.imshow(gradient_img, aspect='auto')
ax.axis('off')  # 축 비활성화

# 여백 최소화 설정
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.gca().set_position([0, 0, 1, 1])  # 그래프를 전체 창에 맞게 조정
plt.margins(0, 0)  # 마진 제거
plt.gca().xaxis.set_major_locator(plt.NullLocator())  # x축 눈금 제거
plt.gca().yaxis.set_major_locator(plt.NullLocator())  # y축 눈금 제거

# 결과 이미지 저장 (원본 이미지 크기와 동일한 크기로 저장)
plt.savefig("sorted_colors_gradient_proportional.png", bbox_inches='tight', pad_inches=0, dpi=orig_width // figsize_width)
plt.show()
