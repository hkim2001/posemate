#mlp_model.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf

# 1. CSV 폴더 경로
csv_folder = "/content/mlp_csv"  # 전처리된 csv 파일 저장 폴더 경로

# 2. CSV 파일 모두 통합
all_df = []
for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(csv_folder, file))
        all_df.append(df)

df_all = pd.concat(all_df, ignore_index=True)

df_all = df_all.dropna(subset=["label"])  # label이 NaN인 행 제거

# 3. Label Encoding (문자 → 숫자)
df_all["label"] = df_all["label"].str.strip()  # 공백 제거
le = LabelEncoder()
df_all["label_encoded"] = le.fit_transform(df_all["label"])  # no_pose 포함됨

# 4. Feature, Label 분리
X = df_all.drop(columns=["label", "filename", "label_encoded"]).values.astype(np.float32)
y = df_all["label_encoded"].values.astype(np.int32)

# 5. 훈련/검증 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 6. TensorFlow 모델 정의 (MLP 구조)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(34,)),           #17 keypoints × (y,x) = 34
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # 드롭아웃 추가
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # 드롭아웃 추가
    tf.keras.layers.Dense(16, activation='relu'),  # 추가된 은닉층
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')  # 클래스 개수 (no_pose 포함)
])




model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7. 학습
model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_data=(X_val, y_val))

# 8. 검증 결과 출력
y_pred = np.argmax(model.predict(X_val), axis=1)

# 실제 있는 클래스만 사용
present_labels = unique_labels(y_val, y_pred)
report = classification_report(
    y_val, y_pred,
    labels=present_labels,
    target_names=le.inverse_transform(present_labels),
    output_dict=True
)

# 9. 분류 리포트 출력 (백분율로 변환)
print("\n 분류 리포트 (백분율 기준)")
for label in le.classes_:
    metrics = report[label]
    print(f"{label:<10}  Precision: {metrics['precision']*100:.1f}%  "
          f"Recall: {metrics['recall']*100:.1f}%  F1-score: {metrics['f1-score']*100:.1f}%  "
          f"Samples: {metrics['support']}개")

# 10. 전체 성능 출력
print(f"\n전체 정확도 (Accuracy): {report['accuracy']*100:.1f}%")
print(f"Macro 평균 F1-score: {report['macro avg']['f1-score']*100:.1f}%")
print(f"Weighted 평균 F1-score: {report['weighted avg']['f1-score']*100:.1f}%")

# 11. 데이터 개수 출력
total = len(df_all)
train_count = len(X_train)
val_count = len(X_val)
print(f"\n 데이터셋 구성")
print(f"학습 데이터: {train_count}개 ({train_count/total*100:.1f}%)")
print(f"검증 데이터: {val_count}개 ({val_count/total*100:.1f}%)")
print(f"전체 데이터: {total}개 (100%)")

# 12. 모델 저장 (Keras 모델)
model.save("mlp_pose_model.h5")

# 13. TFLite 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 14. 저장
with open("mlp_pose_model.tflite", "wb") as f:
    f.write(tflite_model)

print("\nTFLite 모델 저장 완료: mlp_pose_model.tflite")

print("\n 클래스 인코딩 매핑")
for label, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {code}: {label}")
