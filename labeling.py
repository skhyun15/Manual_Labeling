import numpy as np
import cv2

# .npy 파일 로드
npy_file = "C:/Users/chona/Manual_Labeling/0021236.npy"
data = np.load(npy_file)

# 데이터 정규화 (0~255 범위로)
normalized_data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(
    np.uint8
)

# 이미지 저장
output_image = "output_image.png"
cv2.imwrite(output_image, normalized_data)

print(f"이미지 저장 완료: {output_image}")
