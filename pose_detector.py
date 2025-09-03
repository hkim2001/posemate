# pose_cam.py
import cv2
import numpy as np
import argparse
import time
import sys
from tflite_runtime.interpreter import Interpreter

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]

def preprocess_keypoints_runtime(kp):
    kp = np.array(kp)
    center_x = (kp[11][1] + kp[12][1]) / 2
    center_y = (kp[11][0] + kp[12][0]) / 2
    center = np.array([center_y, center_x])
    rel_coords = kp[:, :2] - center
    nose_y = kp[0][0]
    ankle_y = (kp[15][0] + kp[16][0]) / 2
    height = max(abs(ankle_y - nose_y), 1e-5)
    norm_coords = rel_coords / height
    return norm_coords.flatten().astype(np.float32).reshape(1, 34)

def open_usb_camera(device_index: int, width: int = 640, height: int = 480, fps: int = 30):
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"USB 카메라를 열 수 없습니다 (index={device_index}). /dev/video{device_index} 확인.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    def _set_fourcc(fourcc_str: str) -> bool:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
        real_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        chk = "".join([chr((real_fourcc >> 8*i) & 0xFF) for i in range(4)])
        return (chk == fourcc_str)

    chosen = None
    for try_fourcc in ("MJPG", "YUYV"):
        if _set_fourcc(try_fourcc):
            chosen = try_fourcc
            break
    if chosen is None:
        real_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        chosen = "".join([chr((real_fourcc >> 8*i) & 0xFF) for i in range(4)])

    real_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps= cap.get(cv2.CAP_PROP_FPS)
    fps_out = int(real_fps) if real_fps > 0 else -1
    print(f"[V4L2] Using {real_w}x{real_h} @ {fps_out} | FOURCC={chosen}", file=sys.stderr)

    return cap, chosen

def main(is_subprocess: bool,
         device_index: int,
         min_conf: float = 0.25,
         warmup: int = 10,
         verbose: bool = False,
         display_scale: float = 1.0,
         display_width: int = 0,
         display_height: int = 0,
         fullscreen: bool = False):

    movenet = Interpreter(model_path="movenet_lightning.tflite", num_threads=4)
    movenet.allocate_tensors()
    movenet_input = movenet.get_input_details()
    movenet_output = movenet.get_output_details()

    mlp = Interpreter(model_path="mlp_pose_model.tflite", num_threads=2)
    mlp.allocate_tensors()
    mlp_input = mlp.get_input_details()
    mlp_output = mlp.get_output_details()

    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    cap, chosen_fourcc = open_usb_camera(device_index, 640, 480, 30)
    time.sleep(0.5)
    print("USB 카메라 시작됨.", file=sys.stderr)

    WINDOW_NAME = "Pose Classification (USB Cam)"
    if not is_subprocess:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif display_width > 0 and display_height > 0:
            cv2.resizeWindow(WINDOW_NAME, display_width, display_height)

    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if chosen_fourcc == "YUYV":
                if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 2):
                    bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                else:
                    bgr = frame
            else:
                bgr = frame

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (192, 192))
            input_img = np.expand_dims(img.astype(np.uint8), axis=0)

            movenet.set_tensor(movenet_input[0]['index'], input_img)
            movenet.invoke()
            keypoints = movenet.get_tensor(movenet_output[0]['index'])[0][0]

            frame_count += 1
            if frame_count <= warmup:
                label_name = "no_detection"
                print(label_name, flush=True)
                if not is_subprocess:
                    disp = bgr
                    cv2.putText(disp, f"Posture: {label_name}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    disp = resize_display(disp, display_scale, display_width, display_height, fullscreen)
                    cv2.imshow(WINDOW_NAME, disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            mean_conf = float(np.mean(keypoints[:, 2]))
            if mean_conf < min_conf:
                label_name = "no_detection"
            else:
                flat_input = preprocess_keypoints_runtime(keypoints)
                mlp.set_tensor(mlp_input[0]['index'], flat_input)
                mlp.invoke()
                pred = mlp.get_tensor(mlp_output[0]['index'])[0]
                label_idx = int(np.argmax(pred))
                label_name = labels[label_idx]

            print(label_name, flush=True)

            if is_subprocess:
                time.sleep(0.06)
            else:
                disp = bgr.copy()
                points = []
                h, w, _ = disp.shape
                for idx, (y, x, score) in enumerate(keypoints):
                    cx, cy = int(x * w), int(y * h)
                    points.append((idx, (cx, cy) if score > 0.3 else None))

                for a, b in POSE_CONNECTIONS:
                    pt1, pt2 = points[a][1], points[b][1]
                    if pt1 and pt2:
                        cv2.line(disp, pt1, pt2, (255, 255, 255), 2)

                for _, pt in points:
                    if pt:
                        cv2.circle(disp, pt, 4, (0, 255, 0), -1)

                cv2.putText(disp, f"Posture: {label_name}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                disp = resize_display(disp, display_scale, display_width, display_height, fullscreen)
                cv2.imshow(WINDOW_NAME, disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if not is_subprocess:
            cv2.destroyAllWindows()
        print("카메라 종료됨.", file=sys.stderr)

def resize_display(frame, scale, width, height, fullscreen):
    if fullscreen:
        return frame
    elif width > 0 and height > 0:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    elif scale != 1.0:
        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subprocess', action='store_true',
                        help='무창 모드: 라벨만 stdout으로 출력')
    parser.add_argument('--device', type=int, default=1,
                        help='USB 카메라 인덱스 (/dev/video{N}). 기본 1')
    parser.add_argument('--min-conf', type=float, default=0.25,
                        help='MoveNet 평균 신뢰도 임계값')
    parser.add_argument('--warmup', type=int, default=10,
                        help='웜업 프레임 수')
    parser.add_argument('--verbose', action='store_true',
                        help='디버그 로그 출력')
    parser.add_argument('--display-scale', type=float, default=1.0,
                        help='배율 확대 (기본 1.0=원본)')
    parser.add_argument('--display-width', type=int, default=0,
                        help='고정 가로 픽셀')
    parser.add_argument('--display-height', type=int, default=0,
                        help='고정 세로 픽셀')
    parser.add_argument('--fullscreen', action='store_true',
                        help='전체 화면 모드')

    args = parser.parse_args()
    main(args.subprocess, args.device,
         args.min_conf, args.warmup, args.verbose,
         args.display_scale, args.display_width, args.display_height, args.fullscreen)
