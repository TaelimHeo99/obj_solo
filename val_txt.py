import os

images_dir = "datasets/coco/images/val2017"
labels_dir = "datasets/coco/labels/val2017"
output_txt = "datasets/coco/val2017.txt"
missing_txt = "datasets/coco/missing_val_images.txt"

valid_images = []
new_empty_labels = []  # 새로 생성된 빈 라벨 기록

os.makedirs(labels_dir, exist_ok=True)  # 라벨 디렉토리 없으면 생성

for img_name in sorted(os.listdir(images_dir)):
    if not img_name.endswith((".jpg", ".jpeg", ".png")):
        continue
    img_id = os.path.splitext(img_name)[0]
    label_file = os.path.join(labels_dir, img_id + ".txt")

    # ✅ 라벨이 없으면 빈 txt 생성
    if not os.path.exists(label_file):
        open(label_file, "w").close()
        new_empty_labels.append(label_file)

    valid_images.append(f"{images_dir}/{img_name}")

# ✅ 파일 저장
with open(output_txt, "w") as f:
    f.write("\n".join(valid_images))

with open(missing_txt, "w") as f:
    f.write("\n".join(new_empty_labels))  # 새로 생성된 빈 라벨 목록 저장

# ✅ 결과 출력
print(f"✅ 총 이미지 수: {len(valid_images)}")
print(f"🆕 새로 생성된 빈 라벨 수: {len(new_empty_labels)}")
print(f"📂 '{output_txt}' 업데이트 완료.")
print(f"📂 '{missing_txt}'에 새로 생성된 빈 라벨 목록 저장됨.")
