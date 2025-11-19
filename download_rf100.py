# download_rf100.py
#
# Usage:
#   python download_rf100.py sign-language-letters 1
#   python download_rf100.py drone-flir 1
#
# RF100 workspace에서 YOLOv5 포맷으로 데이터셋을 받아
# datasets/rf100/<project_name>/ 아래에 저장한다.

import sys
import pathlib
from roboflow import Roboflow

# >>> 여기에 네 PRIVATE API KEY 넣기 (지금 가지고 있는 그 값)
API_KEY = "Fqb0l6iTAXNQpz5Rcbak"  # <- 따옴표 안만 교체


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_rf100.py <project_name> [version]")
        print("Example: python download_rf100.py sign-language-letters 1")
        sys.exit(1)

    project_name = sys.argv[1]
    version = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

    rf = Roboflow(api_key=API_KEY)

    # RF100 공식 workspace 이름: "roboflow-100"
    ws = rf.workspace("roboflow-100")
    project = ws.project(project_name)
    dataset = project.version(version)

    out_dir = pathlib.Path("datasets") / "rf100" / project_name
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading '{project_name}' v{version} to {out_dir} ...")
    dataset.download("yolov5", location=str(out_dir))
    print(f"[DONE] Downloaded to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
