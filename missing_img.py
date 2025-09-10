import os

images_dir = "datasets/coco/images/train2017"
labels_dir = "datasets/coco/labels/train2017"
txt_file = "datasets/coco/train2017.txt"

missing_images = []

with open(txt_file, "r") as f:
    image_paths = f.read().strip().splitlines()

for path in image_paths:
    if not os.path.exists(path):
        missing_images.append(path)

print(f"🔎 전체 이미지 수: {len(image_paths)}")
print(f"❌ 존재하지 않는 이미지 수: {len(missing_images)}")

# 누락 이미지 미리보기 (최대 10개)
for i, m in enumerate(missing_images[:10]):
    print(f"{i+1}. {m}")
