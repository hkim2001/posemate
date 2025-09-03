#mlp_dataset.py

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.lite.python.interpreter import Interpreter

# 설정
image_folder = "image_standing"              # 이미지 폴더 경로
output_csv = "mlp_dataset_standing.csv"           # 저장할 CSV 파일명
model_path = "movenet_lightning.tflite"  # MoveNet 경로

# 사용할 라벨 목록 (소문자 기준)
allowed_labels = ["squat", "lunge", "pushup", "no_pose"]

# MoveNet 모델 로딩
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# 라벨 추출 함수
def extract_label(fname):
    fname = fname.lower()
    for label in allowed_labels:
        if label in fname:
            return label
    return None

# 관절 추출 함수
def detect_pose(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (input_width, input_height))
    input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return keypoints

# 전처리 함수
def preprocess_keypoints(kp, label):
    kp = np.array(kp)  # shape: (17, 3)

    # 중심점 계산: 좌/우 hip의 중간
    center_x = (kp[11][1] + kp[12][1]) / 2
    center_y = (kp[11][0] + kp[12][0]) / 2
    center = np.array([center_y, center_x])
    rel_coords = kp[:, :2] - center  # (17, 2)

    # height 정의
    if label == "pushup":
        # 가로 방향 거리 (손목 - 발목 x 좌표 차이를 height로)
        wrist_x = (kp[9][1] + kp[10][1]) / 2
        ankle_x = (kp[15][1] + kp[16][1]) / 2
        height = abs(wrist_x - ankle_x)
    else:
        # squat + no_pose (코 - 발목 y 좌표 차이를 height로)
        nose_y = kp[0][0]
        ankle_y = (kp[15][0] + kp[16][0]) / 2
        height = abs(ankle_y - nose_y)

    height = max(height, 1e-5)  # 0으로 나누는 것 방지
    norm_coords = rel_coords / height
    return norm_coords.flatten()

# 데이터 처리 시작
data = []
image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

for fname in image_files:
    label = extract_label(fname)
    if label is None:
        print(f"라벨 인식 실패 → 건너뜀: {fname}")
        continue

    img_path = os.path.join(image_folder, fname)
    keypoints = detect_pose(img_path)
    if keypoints is None:
        print(f"이미지 로드 실패 → 건너뜀: {fname}")
        continue

    processed = preprocess_keypoints(keypoints, label)
    row = [label, os.path.splitext(fname)[0]] + processed.tolist()
    data.append(row)

# 저장
columns = ["label", "filename"] + [f"{joint}_{axis}" for joint in range(17) for axis in ["y", "x"]]
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)
print(f"\n저장 완료: {output_csv} (총 {len(df)}개 샘플)")
