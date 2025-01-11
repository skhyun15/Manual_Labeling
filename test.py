from PIL import Image
import numpy as np

# 배열 출력 옵션 설정
np.set_printoptions(threshold=np.inf)  # 출력 크기 제한 해제

# 이미지 파일 경로
image_path = "segmentation_mask.png"

# 이미지 열기
image = Image.open(image_path)

# 이미지 데이터를 NumPy 배열로 변환
image_array = np.array(image)

print(image_array.shape)  # 배열의 크기 확인
print(image_array)  # 배열 출력
