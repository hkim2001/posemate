# imu_seung11.py
import asyncio, struct, math, time, sys, signal, contextlib
import numpy as np
from collections import deque
from typing import Optional
from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.exc import BleakError

import random

# ========================= 1) 설정 =========================
SERVICE_UUID      = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID         = "abcdefab-1234-5678-1234-56789abcdef1"  # ESP32 -> Pi (Notify)
WRITE_CHAR_UUID   = "abcdefab-1234-5678-1234-56789abcdef2"  # Pi -> ESP32 (Write/WriteNR)
ADAPTER           = "hci0"

DEVICE_ADDRESSES = {
    "ESP32-IMU-WAIST": "0C:4E:A0:32:7E:02",
    "ESP32-IMU-ARM":   "0C:4E:A0:31:C2:DE",
    "ESP32-IMU-THIGH": "0C:4E:A0:66:39:EE",
}
DEVICE_NAMES = list(DEVICE_ADDRESSES.keys())

# 운동 라벨: 1=스쿼트, 2=런지, 3=푸시업
EXERCISE_LABEL = 0
REQUIRED_BY_LABEL = {
    1: ["ESP32-IMU-THIGH", "ESP32-IMU-WAIST"],
    2: ["ESP32-IMU-THIGH", "ESP32-IMU-WAIST"],
    3: ["ESP32-IMU-ARM",   "ESP32-IMU-WAIST"],
}
def required_names():
    return REQUIRED_BY_LABEL.get(EXERCISE_LABEL, [])

# ====== (추가) 평가 게이트 ======
EVAL_ENABLED = True  # 메인 상태머신이 WAIT_NOPOSE/OFFSET_CAL에서 False로 내림

# 샘플링/윈도우
FS = 100
WIN_SHORT  = int(0.3 * FS)   # 0.3 s
HOP_SHORT  = int(0.1 * FS)   # 0.1 s

# 임계값들
GYRO_THRESH        = 30.0
WAIST_PITCH_MAX_OK = 35.0
WAIST_ROLL_ABS_OK  = 30.0

PUSHUP_RANGE_MIN   = 9
PUSHUP_BOTTOM_DEG  = -5.0
MIN_AMPLITUDE_DEG  = 10.0

WAIST_BUM_DOWN_MEAN = -40
WAIST_BUM_UP_MEAN   = -50

SQUAT_OK_MIN_DEG   = -20.0
SQUAT_TOO_DEEP_DEG = -32.0
SQUAT_NOT_DEEP_DEG = -15.0

LUNGE_OK_MIN_DEG   = -20.0
LUNGE_TOO_DEEP_DEG = -38.0
LUNGE_NOT_DEEP_DEG = -15.0

COUNT_MIN_INTERVAL_S = 0.6

# 전처리/필터
OFFSET_MS = 2000
CALIB_MOVE_DPS = 10.0
CALIB_MSG_COOLDOWN = 1.5

FEEDBACK_COOLDOWN_S = 4.0
STATUS_EVERY_S      = 1.0
OK_HUD_EVERY_S      = 1.0
OK_MSG_INTERVAL     = 15.0

ACCEL_SCALE = 16384.0
GYRO_SCALE  = 131.0
FC_ACC      = 5.0
FC_GYRO     = 8.0
ALPHA       = 0.98
DT_MAX_CLAMP = 0.1
EPS          = 1e-8

PKT_STRUCT = struct.Struct("<hhhhhhI")  # 16 bytes: ax ay az gx gy gz t_ms

# 게이팅
DEPTH_OK_LOCKOUT_S = 0.6
SLOPE_N = 5

# ========================= 2) 전역 상태 =========================
states = {}
_last_msg_ts = {}
_last_ok_msg_ts = 0.0
_last_ok_hud_ts = 0.0
_detection_started = False

_connected_devices = set()
ble_clients: dict[str, BleakClient] = {}  # name -> client

_last_ready_debug_ts = 0.0
_connect_lock = None
_connecting: set[str] = set()
_last_seen_label = EXERCISE_LABEL

# 재연결 backoff
RETRY_BASE_DELAY = 1.5
RETRY_CAP        = 15.0

def ensure_state(name: str):
    if name in states: return
    states[name] = {
        'connected': False,
        'start_ms': None, 'last_ms': None,
        'offset_buf': [], 'offsets': None, 'inited': False,
        'facc': {'ax':0.0,'ay':0.0,'az':0.0},
        'fgyro':{'gx':0.0,'gy':0.0,'gz':0.0},
        'pitch': None, 'roll': None,
        'buf_pitch': deque(maxlen=WIN_SHORT),
        'buf_roll':  deque(maxlen=WIN_SHORT),
        'buf_gyro':  deque(maxlen=WIN_SHORT),
        'hop': 0, 'gyro_exceed': 0,
        'last_status_ts': 0.0,
        'calib_started': False, 'last_calib_msg_ts': 0.0,
        'last_depth_ok_ts': 0.0,
    }

# ========================= 3) 유틸/필터 =========================
def lpf_alpha(fc_hz, dt_s):
    if dt_s <= 0: return 1.0
    rc = 1.0/(2*math.pi*fc_hz)
    return dt_s/(rc+dt_s)

def lpf_step(prev, x, alpha):
    return x if prev is None else (prev + alpha*(x-prev))

def accel_to_angles(ax, ay, az):
    pitch = math.degrees(math.atan2(ay, math.sqrt(ax*ax + az*az) + EPS))
    roll  = math.degrees(math.atan2(ax, math.sqrt(ay*ay + az*az) + EPS))
    return pitch, roll

def gyro_rms_of(name):
    st = states[name]
    if len(st['buf_gyro']) < WIN_SHORT: return None
    arr = np.array(st['buf_gyro'])
    return float(np.sqrt(np.mean(arr*arr)))

def device_ready_for_eval(name):
    st = states.get(name)
    return (
        st is not None and st['connected'] and
        st['offsets'] is not None and st['inited'] and
        len(st['buf_pitch']) >= WIN_SHORT
    )

def any_ready():
    return any(device_ready_for_eval(n) for n in states.keys())

def all_required_ready():
    req = required_names()
    return bool(req) and all(device_ready_for_eval(n) for n in req)

def can_emit(key, now_ts, cooldown=FEEDBACK_COOLDOWN_S):
    t0 = _last_msg_ts.get(key, 0.0)
    if now_ts - t0 >= cooldown:
        _last_msg_ts[key] = now_ts
        return True
    return False

async def send_vibration(device_name, duration_s=0.3):
    client = ble_clients.get(device_name)
    if not client or not client.is_connected: return
    try:
        await client.write_gatt_char(WRITE_CHAR_UUID, b'1', response=False)
        await asyncio.sleep(duration_s)
        await client.write_gatt_char(WRITE_CHAR_UUID, b'0', response=False)
    except Exception as e:
        print(f"[{device_name}] 진동 명령 실패: {e}")



# ========================= 4) 평가/메시지 =========================
def arrays_of(dev):
    st = states.get(dev)
    if not st or len(st['buf_pitch']) < WIN_SHORT: return None, None, None
    return (np.array(st['buf_pitch']),
            np.array(st['buf_roll']),
            np.array(st['buf_gyro']))

def compute_ok_status():
    if not all_required_ready():
        return False, "부분모드"
    gyro_vals = [gyro_rms_of(d) for d in states.keys()]
    gyro_vals = [v for v in gyro_vals if v is not None]
    gyro_rms  = min(gyro_vals) if gyro_vals else None
    gyro_ok   = (gyro_rms is not None and gyro_rms <= GYRO_THRESH)

    if EXERCISE_LABEL in (1,2):
        thigh_p, _, _ = arrays_of("ESP32-IMU-THIGH")
        waist_p, waist_r, _ = arrays_of("ESP32-IMU-WAIST")
        if thigh_p is None or waist_p is None or waist_r is None: return False, "부분모드"

        depth_min = float(np.min(thigh_p))
        ok_min_thr = SQUAT_OK_MIN_DEG if EXERCISE_LABEL == 1 else LUNGE_OK_MIN_DEG
        depth_ok   = (depth_min <= ok_min_thr)

        waist_pitch_max   = float(np.max(waist_p))
        waist_roll_absmax = float(np.max(np.abs(waist_r)))
        waist_pitch_ok = (waist_pitch_max   <= WAIST_PITCH_MAX_OK)
        waist_roll_ok  = (waist_roll_absmax <= WAIST_ROLL_ABS_OK)

        all_ok = depth_ok and waist_pitch_ok and waist_roll_ok and gyro_ok
        txt = (f"depth_min {depth_min:.1f}°/≤{ok_min_thr:.0f}° | "
               f"waist_pitch_max {waist_pitch_max:.1f}°/≤{WAIST_PITCH_MAX_OK:.0f}° | "
               f"waist_roll_absmax {waist_roll_absmax:.1f}°/≤{WAIST_ROLL_ABS_OK:.0f}° | "
               f"gyroRMS {gyro_rms:.1f}/≤{GYRO_THRESH:.0f}")
        return all_ok, txt
    else:
        arm_p, _, _   = arrays_of("ESP32-IMU-ARM")
        waist_p, _, _ = arrays_of("ESP32-IMU-WAIST")
        if arm_p is None or waist_p is None: return False, "부분모드"

        range_arm = float(np.max(arm_p) - np.min(arm_p))
        range_ok  = (range_arm >= PUSHUP_RANGE_MIN)
        waist_pitch_mean = float(np.mean(waist_p))
        waist_ok = (waist_pitch_mean <= WAIST_BUM_DOWN_MEAN)
        all_ok = range_ok and waist_ok and gyro_ok
        txt = (f"arm_range {range_arm:.1f}°/≥{PUSHUP_RANGE_MIN:.0f}° | "
               f"waist_pitch_mean {waist_pitch_mean:.1f}°/≤{WAIST_BUM_DOWN_MEAN:.0f}° | "
               f"gyroRMS {gyro_rms:.1f}/≤{GYRO_THRESH:.0f}")
        return all_ok, txt

def partial_feedback():
    if not EVAL_ENABLED:
        return None
    now = time.time()
    msgs = []
    vibrate_waist = False

    st_thigh = states.get("ESP32-IMU-THIGH")
    if st_thigh and device_ready_for_eval("ESP32-IMU-THIGH"):
        thigh_p = np.array(st_thigh['buf_pitch'])
        if len(thigh_p) >= WIN_SHORT:
            depth_min = float(np.min(thigh_p))
            not_deep_thr = SQUAT_NOT_DEEP_DEG if EXERCISE_LABEL == 1 else LUNGE_NOT_DEEP_DEG
            too_deep_thr = SQUAT_TOO_DEEP_DEG if EXERCISE_LABEL == 1 else LUNGE_TOO_DEEP_DEG
            moving_down = False
            if len(thigh_p) >= SLOPE_N + 1:
                moving_down = (np.mean(np.diff(thigh_p[-SLOPE_N:])) < 0)
            if moving_down and depth_min > not_deep_thr and can_emit("p:깊이부족", now):
                msgs.append("조금 더 내려가세요.")
            if depth_min < too_deep_thr and can_emit("p:너무깊음", now):
                msgs.append("무릎을 너무 구부렸어요.")


    st_waist = states.get("ESP32-IMU-WAIST")
    if st_waist and device_ready_for_eval("ESP32-IMU-WAIST"):
        waist_p = np.array(st_waist['buf_pitch'])
        waist_r = np.array(st_waist['buf_roll'])
        if len(waist_p) >= WIN_SHORT:
            if np.max(waist_p) > WAIST_PITCH_MAX_OK and can_emit("p:숙임", now):
                msgs.append("상체가 너무 숙여졌어요!"); vibrate_waist = True
            if np.max(np.abs(waist_r)) > WAIST_ROLL_ABS_OK and can_emit("p:기울임", now):
                msgs.append("상체가 좌우로 기울었어요!"); vibrate_waist = True
            if EXERCISE_LABEL == 3:
                waist_pitch_mean = float(np.mean(waist_p))
                if waist_pitch_mean > WAIST_BUM_DOWN_MEAN and can_emit("p:엉덩이", now):
                    msgs.append("엉덩이를 올리세요"); vibrate_waist = True
                if waist_pitch_mean < WAIST_BUM_UP_MEAN and can_emit("p:엉덩이_위", now):
                    msgs.append("엉덩이를 내리세요"); vibrate_waist = True

    st_arm = states.get("ESP32-IMU-ARM")
    if st_arm and device_ready_for_eval("ESP32-IMU-ARM") and EXERCISE_LABEL == 3:
        arm_p = np.array(st_arm['buf_pitch'])
        if len(arm_p) >= WIN_SHORT:
            arm_range = float(np.max(arm_p) - np.min(arm_p))
            if arm_range < PUSHUP_RANGE_MIN and can_emit("p:팔범위", now):
                msgs.append("팔을 조금 더 굽히세요!")

    if msgs and vibrate_waist:
        asyncio.create_task(send_vibration("ESP32-IMU-WAIST"))
    return msgs[0] if msgs else None

def posture_feedback_full():
    if not EVAL_ENABLED:
        return None
    now = time.time()
    msgs = []
    vibrate_waist = False

    thigh = states.get("ESP32-IMU-THIGH")
    waist = states.get("ESP32-IMU-WAIST")
    arm   = states.get("ESP32-IMU-ARM")

    def arr(st, key):
        if not st or len(st['buf_pitch']) < WIN_SHORT: return None
        if key == 'pitch': return np.array(st['buf_pitch'])
        if key == 'roll' : return np.array(st['buf_roll'])
        return None

    if EXERCISE_LABEL in (1,2):
        thigh_p = arr(thigh,'pitch'); waist_p = arr(waist,'pitch'); waist_r = arr(waist,'roll')
        if thigh_p is not None:
            depth_min = float(np.min(thigh_p))
            not_deep_thr = SQUAT_NOT_DEEP_DEG if EXERCISE_LABEL == 1 else LUNGE_NOT_DEEP_DEG
            too_deep_thr = SQUAT_TOO_DEEP_DEG if EXERCISE_LABEL == 1 else LUNGE_TOO_DEEP_DEG
            lockout_active = (time.time() - thigh['last_depth_ok_ts']) < DEPTH_OK_LOCKOUT_S
            moving_down = False
            if len(thigh_p) >= SLOPE_N + 1:
                moving_down = (np.mean(np.diff(thigh_p[-SLOPE_N:])) < 0)
            if (not lockout_active) and moving_down and (depth_min > not_deep_thr) and can_emit("깊이부족", now):
                msgs.append("조금 더 내려가세요.")
            if depth_min < too_deep_thr and can_emit("너무깊음", now):
                msgs.append("무릎을 너무 구부렸어요.")
        if waist_p is not None and np.max(waist_p) > WAIST_PITCH_MAX_OK and can_emit("숙임", now):
            msgs.append("상체가 너무 숙여졌어요!"); vibrate_waist = True
        if waist_r is not None and np.max(np.abs(waist_r)) > WAIST_ROLL_ABS_OK and can_emit("기울임", now):
            msgs.append("상체가 좌우로 기울었어요!"); vibrate_waist = True
    else:
        arm_p   = arr(arm,'pitch'); waist_p = arr(waist,'pitch')
        if arm_p is not None:
            arm_range = float(np.max(arm_p) - np.min(arm_p))
            if arm_range < PUSHUP_RANGE_MIN and can_emit("팔굽힘범위", now):
                msgs.append("팔을 조금 더 굽히세요!")
        if waist_p is not None:
            waist_pitch_mean = float(np.mean(waist_p))
            if waist_pitch_mean > WAIST_BUM_DOWN_MEAN and can_emit("엉덩이", now):
                msgs.append("엉덩이를 올리세요"); vibrate_waist = True
            if waist_pitch_mean < WAIST_BUM_UP_MEAN and can_emit("엉덩이_위", now):
                msgs.append("엉덩이를 내리세요"); vibrate_waist = True

    if msgs and vibrate_waist:
        asyncio.create_task(send_vibration("ESP32-IMU-WAIST"))
    return msgs[0] if msgs else None

# ========================= 5) BLE 보조: 서비스/특성 탐색 =========================
async def resolve_characteristics(name: str, client: BleakClient):
    n_uuid, w_uuid = None, None
    try:
        try:
            services = await client.get_services()
        except TypeError:
            services = client.services or client.get_services()
    except Exception as e:
        print(f"[경고] {name} 서비스 조회 실패: {e}")
        return None, None

    try:
        for svc in services:
            if str(svc.uuid).lower() == SERVICE_UUID.lower():
                for ch in svc.characteristics:
                    props = set(ch.properties or [])
                    if 'notify' in props and n_uuid is None:
                        n_uuid = str(ch.uuid)
                    if ('write' in props or 'write-without-response' in props) and w_uuid is None:
                        w_uuid = str(ch.uuid)
    except Exception as e:
        print(f"[경고] {name} 서비스 파싱 실패: {e}")
    return n_uuid, w_uuid

# ========================= 5) BLE 수신/연결 =========================
def parse_many_samples(data: bytes):
    if len(data) % PKT_STRUCT.size != 0:
        return []
    n = len(data) // PKT_STRUCT.size
    return [PKT_STRUCT.unpack_from(data, i * PKT_STRUCT.size) for i in range(n)]

def make_notify_cb(name):
    ensure_state(name)
    st = states[name]

    async def on_notify(_, data: bytearray):
        if name not in required_names():
            return

        samples = parse_many_samples(data)
        if not samples:
            return

        for (ax, ay, az, gx, gy, gz, t_ms) in samples:
            st['connected'] = True

            ax_g, ay_g, az_g = ax/ACCEL_SCALE, ay/ACCEL_SCALE, az/ACCEL_SCALE
            gx_dps, gy_dps, gz_dps = gx/GYRO_SCALE, gy/GYRO_SCALE, gz/GYRO_SCALE

            if st['start_ms'] is None:
                st['start_ms'] = t_ms
            elapsed = t_ms - st['start_ms']

            if st['offsets'] is None:
                now_ts = time.time()
                if not st['calib_started']:
                    st['calib_started'] = True
                    print(f"[{name}] 오프셋 보정 시작: 정자세로 1~2초 가만히 계세요.")
                gyro_mag_inst = float(np.linalg.norm([gx_dps, gy_dps, gz_dps]))
                if gyro_mag_inst > CALIB_MOVE_DPS and can_emit(f"{name}:calib_move", now_ts, CALIB_MSG_COOLDOWN):
                    print(f"[{name}] 움직임 감지: 잠시 가만히 계세요.")
                st['offset_buf'].append((ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps))
                if elapsed >= OFFSET_MS:
                    m = np.mean(np.array(st['offset_buf']), axis=0)
                    st['offsets'] = dict(zip(['ax','ay','az','gx','gy','gz'], m))
                    print(f"[{name}] 보정 완료: 이제 움직이세요.")
                continue

            ax_c = ax_g - st['offsets']['ax']
            ay_c = ay_g - st['offsets']['ay']
            az_c = az_g
            gx_c = gx_dps - st['offsets']['gx']
            gy_c = gy_dps - st['offsets']['gy']
            gz_c = gz_dps - st['offsets']['gz']

            dt = 0.01 if st['last_ms'] is None else (t_ms - st['last_ms'])/1000.0
            st['last_ms'] = t_ms
            if dt < 0: dt = 0.0
            if dt > DT_MAX_CLAMP: dt = DT_MAX_CLAMP

            if not st['inited']:
                st['facc']['ax'], st['facc']['ay'], st['facc']['az'] = ax_c, ay_c, az_c
                st['fgyro']['gx'], st['fgyro']['gy'], st['fgyro']['gz'] = gx_c, gy_c, gz_c
                p0, r0 = accel_to_angles(st['facc']['ax'], st['facc']['ay'], st['facc']['az'])
                st['pitch'], st['roll'] = p0, r0
                st['inited'] = True
                continue

            a_acc, a_gyro = lpf_alpha(FC_ACC, dt), lpf_alpha(FC_GYRO, dt)
            st['facc']['ax'] = lpf_step(st['facc']['ax'], ax_c, a_acc)
            st['facc']['ay'] = lpf_step(st['facc']['ay'], ay_c, a_acc)
            st['facc']['az'] = lpf_step(st['facc']['az'], az_c, a_acc)
            st['fgyro']['gx'] = lpf_step(st['fgyro']['gx'], gx_c, a_gyro)
            st['fgyro']['gy'] = lpf_step(st['fgyro']['gy'], gy_c, a_gyro)
            st['fgyro']['gz'] = lpf_step(st['fgyro']['gz'], gz_c, a_gyro)

            p_acc, r_acc = accel_to_angles(st['facc']['ax'], st['facc']['ay'], st['facc']['az'])
            p_gyro = st['pitch'] + st['fgyro']['gy'] * dt
            r_gyro = st['roll']  + st['fgyro']['gx'] * dt
            st['pitch'] = ALPHA * p_gyro + (1-ALPHA) * p_acc
            st['roll']  = ALPHA * r_gyro + (1-ALPHA) * r_acc

            gyro_mag = float(np.linalg.norm([st['fgyro']['gx'], st['fgyro']['gy'], st['fgyro']['gz']]))
            st['buf_pitch'].append(st['pitch'])
            st['buf_roll'].append(st['roll'])
            st['buf_gyro'].append(gyro_mag)
            st['hop'] += 1

            if EXERCISE_LABEL in (1, 2) and name == "ESP32-IMU-THIGH":
                ok_min_thr = SQUAT_OK_MIN_DEG if EXERCISE_LABEL == 1 else LUNGE_OK_MIN_DEG
                if st['pitch'] <= ok_min_thr:
                    states[name]['last_depth_ok_ts'] = time.time()

            now_ts = time.time()
            if now_ts - st['last_status_ts'] >= STATUS_EVERY_S:
                st['last_status_ts'] = now_ts
                rms = gyro_rms_of(name)
                rms_txt = f"{rms:5.1f} dps" if rms is not None else "  ---  "
                print(f"=================== \n[{name}] pitch {st['pitch']:6.1f}° | roll {st['roll']:6.1f}° | gyroRMS(0.3s) {rms_txt}  \n===================\n" )
    return on_notify

# --------- (추가) 오프셋 재보정 유틸 ---------
def reset_offsets_for_required():
    """현재 라벨에 필요한 센서들의 오프셋/버퍼/초기화 상태를 리셋하여 OFFSET 보정 재시작."""
    req = required_names()
    now = time.time()
    for name in req:
        ensure_state(name)
        st = states[name]
        st['start_ms'] = None
        st['last_ms'] = None
        st['offset_buf'].clear()
        st['offsets'] = None
        st['inited'] = False
        st['buf_pitch'].clear()
        st['buf_roll'].clear()
        st['buf_gyro'].clear()
        st['calib_started'] = False
        st['last_calib_msg_ts'] = now
    print(f"[매니저] OFFSET 재보정 준비 완료(센서: {', '.join(req)})")

# --------- 연결/해제 유틸 ---------
def _is_connected(name: str) -> bool:
    c = ble_clients.get(name)
    return bool(c and c.is_connected)

async def _disconnect_device(name: str):
    client = ble_clients.pop(name, None)
    states[name]['connected'] = False
    _connected_devices.discard(name)
    if client and client.is_connected:
        with contextlib.suppress(Exception):
            await client.disconnect()
        print(f"[연결해제] {name} 해제 완료")

# --------- 연결 시도 (무한 재시도 백오프) ---------
async def connect_one(name: str, address: str):
    ensure_state(name)
    attempt = 0
    while True:
        attempt += 1
        delay = min(RETRY_CAP, RETRY_BASE_DELAY * (2 ** max(0, attempt - 1)))
        try:
            async with _connect_lock:
                print(f"[연결] {name} 스캔/발견 시도 ... (attempt={attempt})")
                dev: Optional[BLEDevice] = await BleakScanner.find_device_by_address(
                    address, timeout=8.0, adapter=ADAPTER
                )
            if dev is None:
                print(f"[경고] {name}({address}) 미발견 - {delay:.1f}s 후 재시도")
                await asyncio.sleep(delay); continue

            client = BleakClient(dev, adapter=ADAPTER)
            await client.connect(timeout=12.0)

            try:
                n_uuid, w_uuid = await resolve_characteristics(name, client)
            except Exception as e:
                print(f"[경고] {name} 서비스 탐색 실패, 고정 UUID로 진행: {e}")
                n_uuid, w_uuid = None, None
            notify_uuid = n_uuid or CHAR_UUID

            states[name]['connected'] = True
            _connected_devices.add(name)
            ble_clients[name] = client

            cb = make_notify_cb(name)
            await client.start_notify(notify_uuid, cb)
            print(f"[성공] {name} 연결({address}) | notify={notify_uuid}")

            while True:
                await asyncio.sleep(1.0)
                if not client.is_connected:
                    raise BleakError("disconnected")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            states[name]['connected'] = False
            _connected_devices.discard(name)
            if name in ble_clients and not ble_clients[name].is_connected:
                ble_clients.pop(name, None)
            print(f"[에러] {name} 세션 오류: {e!r} → {delay:.1f}s 후 재시도")
            await asyncio.sleep(delay)

# ========================= 6) 준비상태 디버그 & 중앙 평가 루프 =========================
def debug_ready_status():
    req = required_names()
    names = req if req else DEVICE_NAMES
    lines = []
    for n in names:
        st = states.get(n)
        if not st:
            lines.append(f"- {n}: 상태 없음"); continue
        lines.append(
            f"- {n}: connected={st['connected']} | offsets={'Y' if st['offsets'] is not None else 'N'} | "
            f"inited={'Y' if st['inited'] else 'N'} | buf_pitch={len(st['buf_pitch'])}/{WIN_SHORT}"
        )
    print("[준비/부분모드 상태]\n" + "\n".join(lines))

async def evaluator_loop():
    global _detection_started, _last_ok_hud_ts, _last_ok_msg_ts, _last_ready_debug_ts
    while True:
        await asyncio.sleep(0.1)  # 10 Hz

        if not any(_is_connected(n) for n in DEVICE_NAMES):
            now = time.time()
            if now - _last_ready_debug_ts >= 2.0:
                _last_ready_debug_ts = now
                debug_ready_status()
            continue

        if any_ready():
            if all_required_ready():
                if not _detection_started and EVAL_ENABLED:
                    _detection_started = True
                    req = ", ".join(required_names())
                    label_txt = {1:"스쿼트",2:"런지",3:"푸시업"}[EXERCISE_LABEL]
                    print(f"필수 센서 준비 완료({req}). {label_txt} 감지를 시작합니다.")
                warn = posture_feedback_full()
                if warn: print(warn)

                all_ok, ok_txt = compute_ok_status()
                now_ts = time.time()
                if EVAL_ENABLED and all_ok:
                    if now_ts - _last_ok_hud_ts >= OK_HUD_EVERY_S:
                        _last_ok_hud_ts = now_ts
                        prefix = "스쿼트OK" if EXERCISE_LABEL==1 else ("런지OK" if EXERCISE_LABEL==2 else "푸시업OK")
                        print(f"[{prefix}] {ok_txt}")
                    if now_ts - _last_ok_msg_ts >= OK_MSG_INTERVAL:
                        _last_ok_msg_ts = now_ts
                        print("바른 자세예요!")
            else:
                warn = partial_feedback()
                if warn: print(f"[부분모드] {warn}")
                now = time.time()
                if now - _last_ready_debug_ts >= 1.0:
                    _last_ready_debug_ts = now
                    missing = [n for n in required_names() if not device_ready_for_eval(n)]
                    if missing:
                        print("[부분모드] 준비 안 된 센서: " + ", ".join(missing))
                        debug_ready_status()
        else:
            now = time.time()
            if now - _last_ready_debug_ts >= 1.0:
                _last_ready_debug_ts = now
                debug_ready_status()

# ========================= 7) 디바이스 매니저(라벨 변경 대응) =========================
async def device_manager_loop():
    global _last_seen_label, _detection_started

    async def ensure_required_now():
        req = set(required_names())

        for name in list(ble_clients.keys()):
            if name not in req:
                print(f"[매니저] '{name}'는 현재 운동에 불필요 → 해제")
                await _disconnect_device(name)

        for name in req:
            if not _is_connected(name) and name not in _connecting:
                _connecting.add(name)
                async def _runner(n=name):
                    try:
                        await connect_one(n, DEVICE_ADDRESSES[n])
                    finally:
                        _connecting.discard(n)
                asyncio.create_task(_runner(), name=f"connect:{name}")

    await ensure_required_now()

    while True:
        await asyncio.sleep(1.0)

        if EXERCISE_LABEL != _last_seen_label:
            old = _last_seen_label
            _last_seen_label = EXERCISE_LABEL
            _detection_started = False
            label_txt = {1:"스쿼트",2:"런지",3:"푸시업"}.get(EXERCISE_LABEL, str(EXERCISE_LABEL))
            print(f"[매니저] 라벨 변경 감지: {old} → {EXERCISE_LABEL} ({label_txt}). BLE 재탐색/재연결.")
            await ensure_required_now()
        else:
            await ensure_required_now()

# ========================= 8) 메인/종료 핸들러 =========================
async def main():
    global _connect_lock
    _connect_lock = asyncio.Lock()

    EXERCISE_MAP = {1: "스쿼트", 2: "런지", 3: "푸시업"}
    label_txt = EXERCISE_MAP.get(EXERCISE_LABEL, "알 수 없음")
    print(f"[IMU] 현재 운동 '{label_txt}'에 필요한 센서를 유지합니다(라벨 변경 자동 대응).")

    try:
        await asyncio.gather(
            device_manager_loop(),
            evaluator_loop(),
        )
    except asyncio.CancelledError:
        print("[IMU] 작업 취소 요청 수신.")
    finally:
        print("[IMU] 모든 클라이언트 연결 해제 중...")
        for name in list(ble_clients.keys()):
            await _disconnect_device(name)
        print("[IMU] 모든 클라이언트 연결 해제 완료.")

def _graceful_exit(*_):
    print("\n[종료] SIGINT/SIGTERM 수신")
    try:
        loop = asyncio.get_running_loop()
        for task in asyncio.all_tasks(loop):
            task.cancel()
    except Exception:
        pass
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _graceful_exit()
