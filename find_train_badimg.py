import os
import cv2

image_dir = "datasets/coco/images/train2017"
bad_images = []

for idx, img_name in enumerate(sorted(os.listdir(image_dir))):
    if not img_name.endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ [{idx}] 손상된 이미지: {img_name}")
        bad_images.append(img_path)

# 파일로 저장
with open("bad_images.txt", "w") as f:
    for path in bad_images:
        f.write(path + "\n")

print(f"\n✅ 총 손상된 이미지 수: {len(bad_images)}개")
