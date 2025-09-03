# main.py
import asyncio, sys, re, signal, time, os, contextlib, io
import subprocess
import imu_sensor as imu
from gtts import gTTS

# ====== 환경/설정 ======
AUDIO_DEVICE   = os.environ.get("AUDIO_DEVICE", "plughw:1,0")
SPEED          = float(os.environ.get("TTS_SPEED", "1.25"))  # 0.5 ~ 2.0
LANG           = os.environ.get("TTS_LANG", "ko")

CAM_DEVICE_IDX = int(os.environ.get("CAM_DEVICE", "0"))
MIN_CONF       = os.environ.get("MIN_CONF", "0.25")
WARMUP_FRAMES  = os.environ.get("WARMUP", "10")
VERBOSE_FLAG   = os.environ.get("VERBOSE", "1")

DISPLAY_SCALE  = os.environ.get("DISPLAY_SCALE", "1.5")
DISPLAY_WIDTH  = os.environ.get("DISPLAY_WIDTH", "0")
DISPLAY_HEIGHT = os.environ.get("DISPLAY_HEIGHT", "0")
FULLSCREEN     = os.environ.get("FULLSCREEN", "0")

LABEL_MAP = {"squat": 1, "lunge": 2, "pushup": 3}

# ====== 타이밍(초) ======
T_LOCK = 0.8     # 같은 운동 라벨이 연속 유지되어 ACTIVE 확정되는 시간
T_NP   = 1.2     # ACTIVE 중 no_pose가 연속 유지될 때 OFFSET으로 들어가는 시간
T_OFF  = 1.5     # OFFSET 보정 샘플 수집 시간

# ====== 유틸 ======
def atempo_filter(speed: float) -> str:
    if 0.5 <= speed <= 2.0:
        return f"atempo={speed}"
    chain = []
    s = speed
    while s > 2.0:
        chain.append("atempo=2.0"); s /= 2.0
    while s < 0.5:
        chain.append("atempo=0.5"); s /= 0.5
    chain.append(f"atempo={s}")
    return ",".join(chain)

# ====== gTTS 비동기 스피커 ======
class GTTSpeaker:
    def __init__(self, audio_device=AUDIO_DEVICE, speed=SPEED, lang=LANG):
        self.q = asyncio.Queue()
        self.audio_device = audio_device
        self.speed = speed
        self.lang = lang
        self.cache = {}
        self.last_spoken = {}
        self.cooldown = 2.5

    async def speak(self, text: str, priority: bool = False):
        text = (text or "").strip()
        if not text: return
        await self.q.put((time.time(), text, priority))

    def _synthesize_mp3_bytes(self, text: str) -> bytes:
        buf = io.BytesIO()
        gTTS(text=text, lang=self.lang).write_to_fp(buf)
        return buf.getvalue()

    def _play_bytes_with_ffmpeg_aplay(self, mp3_bytes: bytes):
        af = atempo_filter(self.speed)
        p1 = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-filter:a", af, "-f", "wav", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        p2 = subprocess.Popen(["aplay", "-D", self.audio_device, "-q"], stdin=p1.stdout)
        try:
            if p1.stdin:
                p1.stdin.write(mp3_bytes); p1.stdin.close()
        except BrokenPipeError:
            pass
        p2.wait()

    async def worker(self):
        while True:
            _, text, priority = await self.q.get()
            now = time.time()
            t0 = self.last_spoken.get(text, 0.0)
            if not priority and (now - t0) < self.cooldown:
                continue
            self.last_spoken[text] = now
            try:
                mp3_bytes = self.cache.get(text)
                if mp3_bytes is None:
                    mp3_bytes = await asyncio.to_thread(self._synthesize_mp3_bytes, text)
                    if len(text) <= 100:
                        self.cache[text] = mp3_bytes
                await asyncio.to_thread(self._play_bytes_with_ffmpeg_aplay, mp3_bytes)
            except Exception as e:
                print(f"[TTS] 오류: {e}")

# ====== imu_module 내부 print 후킹 ======
def install_imu_print_router(speaker: GTTSpeaker):
    orig_print = print
    tts_re = re.compile(
        r"(조금 더 내려가세요|무릎을 너무 구부렸어요|엉덩이를\s*(올리세요|내리세요)|"
        r"팔을\s*조금\s*더\s*굽히세요!?|"
        r"상체가 너무 숙여졌어요!|상체가 좌우로 기울었어요!|"
        r"바른 자세예요!|반복\s*\d+\s*회)",
        re.UNICODE
    )
    def routed_print(*args, **kwargs):
        s = " ".join(str(a) for a in args).strip()
        orig_print(*args, **kwargs)
        s_clean = re.sub(r"^\[(부분모드|경고|에러)\]\s*", "", s)
        if tts_re.search(s_clean):
            s_clean = s_clean.replace("|", ", ").replace("°", "도")
            asyncio.create_task(speaker.speak(s_clean))
    imu.print = routed_print

# ====== pose_detector.py 라벨 읽기 (모든 라벨을 그대로 전달) ======
async def run_pose_subproc(label_queue: asyncio.Queue, cam_index: int = CAM_DEVICE_IDX):
    """
    pose_detector.py를 GUI 모드로 실행하여 표준출력 라벨을 읽어 label_queue로 전달.
    ★ 중요: 상태머신에서 T_LOCK/T_NP를 처리하므로, 여기서는 '모든 라벨'을 즉시 전달한다.
    """
    argv = [
        sys.executable, "-u", "pose_detector.py",
        "--device", str(cam_index),
        "--min-conf", str(MIN_CONF),
        "--warmup", str(WARMUP_FRAMES),
    ]
    if VERBOSE_FLAG == "1": argv.append("--verbose")
    if FULLSCREEN == "1":
        argv.append("--fullscreen")
    else:
        argv += ["--display-scale", str(DISPLAY_SCALE),
                 "--display-width", str(DISPLAY_WIDTH),
                 "--display-height", str(DISPLAY_HEIGHT)]

    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    print(f"[메인] pose_detector.py 시작 (device={cam_index}, GUI 모드)")

    async def log_stderr():
        while True:
            line = await proc.stderr.readline()
            if not line: break
            print(f"[PoseCam] {line.decode().strip()}")
    stderr_task = asyncio.create_task(log_stderr())

    try:
        while True:
            line = await proc.stdout.readline()
            if not line: break
            label = line.decode("utf-8").strip()
            await label_queue.put((time.time(), label))  # (timestamp, label)
    finally:
        stderr_task.cancel()
        if proc.returncode is None:
            proc.terminate(); await proc.wait()
        print("[메인] pose_detector.py 종료")

# ====== 상태 머신 ======
class PostureStateMachine:
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    OFFSET = "OFFSET_CAL"

    def __init__(self, speaker: GTTSpeaker, imu_start_event: asyncio.Event):
        self.speaker = speaker
        self.imu_start_event = imu_start_event
        self.state = self.IDLE
        self.active_mode = None          # 1/2/3
        self.lock_label = None           # 현재 락 시도 중 라벨(str)
        self.lock_start_ts = None        # 락 시작 시각
        self.no_pose_start_ts = None     # ACTIVE 중 no_pose 시작 시각
        self.offset_end_ts = None        # OFFSET 종료 예정 시각
        self.first_imu_started = False

        # imu.EVAL_ENABLED 기본값 보정
        if not hasattr(imu, "EVAL_ENABLED"):
            imu.EVAL_ENABLED = False

    # ---- IMU 상호작용 헬퍼 ----
    def _set_eval(self, flag: bool):
        try:
            imu.EVAL_ENABLED = flag
        except Exception:
            pass

    def _apply_mode_to_imu(self, mode_int: int):
        # 안전하게 모드/카운터/필요 센서 전환
        try:
            imu.EXERCISE_LABEL = mode_int
        except Exception:
            pass
        try:
            if hasattr(imu, "set_required_sensors_for_mode"):
                imu.set_required_sensors_for_mode(mode_int)
        except Exception:
            pass
        try:
            if hasattr(imu, "setup_rep_counter"):
                imu.setup_rep_counter()
        except Exception:
            pass

    def _reset_offsets_for_required(self, mode_int: int):
        # 요구된 센서만 오프셋/버퍼 초기화
        if hasattr(imu, "reset_offsets_for_required"):
            try:
                imu.reset_offsets_for_required(mode_int)
            except Exception as e:
                print(f"[메인] reset_offsets_for_required 호출 실패: {e}")
        else:
            print("[메인] reset_offsets_for_required가 없어 전체 초기화를 가정합니다.")
            try:
                if hasattr(imu, "reset_all_offsets"):
                    imu.reset_all_offsets()
            except Exception:
                pass

    # ---- 상태 전이 ----
    async def _enter_active(self, mode_name: str, mode_int: int, priority_tts=True):
        self.state = self.ACTIVE
        self.active_mode = mode_int
        self.no_pose_start_ts = None
        self.lock_label = None; self.lock_start_ts = None

        # IMU 시작(최초 1회) 또는 모드 전환
        self._apply_mode_to_imu(mode_int)
        if not self.first_imu_started:
            self.imu_start_event.set()
            self.first_imu_started = True

        self._set_eval(True)
        await self.speaker.speak(f"{mode_name} 자세가 감지되었습니다.", priority=priority_tts)
        print(f"[상태] ACTIVE({mode_name}) 진입")

    async def _enter_offset(self):
        self.state = self.OFFSET
        self._set_eval(False)   # 즉시 피드백 중단
        await self.speaker.speak("가만히 계세요.", priority=True)
        # 현재 운동 기준으로 오프셋/버퍼 보정 시작
        if self.active_mode is not None:
            self._reset_offsets_for_required(self.active_mode)
        self.offset_end_ts = time.time() + T_OFF
        self.no_pose_start_ts = None
        self.lock_label = None; self.lock_start_ts = None
        print("[상태] OFFSET_CAL 진입 (보정 수집 중)")

    def _enter_idle(self):
        self.state = self.IDLE
        self.active_mode = None
        self.lock_label = None; self.lock_start_ts = None
        self.no_pose_start_ts = None
        self._set_eval(False)  # 새 운동 확정 전까지 계속 OFF
        print("[상태] IDLE 진입")

    # ---- 라벨 처리 ----
    async def handle_label(self, ts: float, label: str):
        label_l = (label or "").lower().strip()
        is_pose = label_l in LABEL_MAP
        now = ts

        # OFFSET 중에는 '어떤 라벨도 확정 금지' - 단, 타이머는 timer_task에서 끝낸다.
        if self.state == self.OFFSET:
            # OFFSET 동안 들어오는 라벨은 전부 무시
            return

        # ACTIVE 상태에서의 no_detection / no_pose 처리
        if self.state == self.ACTIVE:
            # “no_detection”은 전환 영향 없음 + no_pose 타이머 리셋
            if "no_detection" in label_l:
                self.no_pose_start_ts = None
                return
            # “no_pose”가 연속 T_NP 유지되면 OFFSET으로
            if "no_pose" in label_l:
                if self.no_pose_start_ts is None:
                    self.no_pose_start_ts = now
                elif (now - self.no_pose_start_ts) >= T_NP:
                    await self._enter_offset()
                return
            else:
                # 자세가 다시 보이면 no_pose 타이머 리셋
                self.no_pose_start_ts = None

            # 다른 운동으로 바꾸려면 다시 락을 잡아야 함 → IDLE로 내려가 락 시작
            if is_pose and LABEL_MAP[label_l] != self.active_mode:
                self._enter_idle()
                # 아래 IDLE 로직으로 이어져 T_LOCK을 새로 잡는다.

        # IDLE 상태: 운동 라벨이면 T_LOCK 계산
        if self.state == self.IDLE:
            if is_pose:
                if self.lock_label != label_l:
                    # 다른 라벨이 들어오면 락 재시작
                    self.lock_label = label_l
                    self.lock_start_ts = now
                else:
                    # 같은 라벨이 유지되면 경과 확인
                    if self.lock_start_ts and (now - self.lock_start_ts) >= T_LOCK:
                        mode_int = LABEL_MAP[label_l]
                        name = {1: "스쿼트", 2: "런지", 3: "푸시업"}.get(mode_int, "알 수 없는 운동")
                        await self._enter_active(name, mode_int, priority_tts=True)
                return
            else:
                # IDLE에선 no_pose/no_detection은 단순 무시, 락도 리셋
                self.lock_label = None
                self.lock_start_ts = None
                return

    # OFFSET 종료 타이머 체크 (주기적)
    async def timer_task(self):
        try:
            while True:
                await asyncio.sleep(0.05)
                if self.state == self.OFFSET and self.offset_end_ts is not None:
                    if time.time() >= self.offset_end_ts:
                        # OFFSET 종료 → IDLE 복귀 (EVAL_ENABLED=False 유지)
                        self.offset_end_ts = None
                        self._enter_idle()
        except asyncio.CancelledError:
            pass

# ====== 라벨 소비 태스크 ======
async def state_router_task(label_queue: asyncio.Queue, sm: PostureStateMachine):
    while True:
        ts, label = await label_queue.get()
        try:
            await sm.handle_label(ts, label)
        except Exception as e:
            print(f"[상태] 라벨 처리 오류: {e}")

# ====== 메인 오케스트레이션 ======
async def main_orchestrator():
    speaker = GTTSpeaker()
    install_imu_print_router(speaker)
    label_queue = asyncio.Queue()
    imu_start_event = asyncio.Event()

    # 1) TTS, 상태머신 시작
    tts_task = asyncio.create_task(speaker.worker(), name="tts_worker")
    sm = PostureStateMachine(speaker, imu_start_event)
    sm_timer = asyncio.create_task(sm.timer_task(), name="state_timer")
    router_task = asyncio.create_task(state_router_task(label_queue, sm), name="state_router")

    # 2) 포즈 카메라 시작 (모든 라벨 전달)
    print("[메인] Pose Cam을 시작합니다...")
    pose_task = asyncio.create_task(run_pose_subproc(label_queue, CAM_DEVICE_IDX), name="pose_cam")

    # 3) 첫 ACTIVE 확정(=유효 운동)까지 대기 후 IMU 시작
    print("\n[메인] 사용자가 운동 자세를 취하면 IMU 센서 연결을 시작합니다...")
    await imu_start_event.wait()
    print("[메인] IMU 연결 시작!")
    imu_task = asyncio.create_task(imu.main(), name="imu_main")

    # 4) 우아한 종료 처리
    tasks = {tts_task, sm_timer, router_task, pose_task, imu_task}
    stop_event = asyncio.Event()

    def handle_signal():
        print("\n[메인] 종료 신호 수신. 모든 작업을 정리합니다.")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    stop_task = asyncio.create_task(stop_event.wait(), name="stop_event_waiter")
    tasks.add(stop_task)

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    print("[메인] 첫 번째 태스크 종료. 나머지 태스크를 취소합니다.")
    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    print("[메인] 모든 작업이 종료되었습니다.")

if __name__ == "__main__":
    try:
        asyncio.run(main_orchestrator())
    except KeyboardInterrupt:
        print("[메인] 강제 종료")

