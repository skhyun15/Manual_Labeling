import json
import numpy as np
import cv2
from PIL import Image

# Labelme JSON 파일 경로
json_file = "pre_jpg_folder_orig/train/test.json"

# JSON 파일 로드
with open(json_file) as f:
    data = json.load(f)

# 이미지 크기 추출
image_height = data["imageHeight"]
image_width = data["imageWidth"]

# 빈 마스크 생성 (모든 픽셀 0으로 초기화)
mask = np.zeros((image_height, image_width), dtype=np.uint8)

# 클래스 ID를 라벨 이름과 매핑
label_map = {
    "a": 1,
    "b": 2,  # 예: 다리뼈
    # 필요한 클래스 추가
}

# 폴리곤 데이터를 마스크에 채우기
for shape in data["shapes"]:
    label = shape["label"]
    points = np.array(shape["points"], dtype=np.int32)  # 폴리곤 좌표
    class_id = label_map[label]  # 클래스 ID 가져오기
    cv2.fillPoly(mask, [points], class_id)  # 폴리곤 내부를 클래스 ID로 채움
print("Before conversion:", mask.dtype, np.unique(mask))


# 결과 저장
output_mask_file = "segmentation_mask.png"
Image.fromarray(mask.astype(np.uint8)).save(output_mask_file)
print(f"Segmentation mask 저장 완료: {output_mask_file}")


# import matplotlib.pyplot as plt

# plt.imshow(mask, cmap="tab20")
# plt.show()
