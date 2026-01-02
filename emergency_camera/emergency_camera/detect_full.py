import cv2
import time
from ultralytics import YOLO

# =========================================================
# [설정] MJPG 고속 모드 + 예외 처리 통합
# =========================================================
MODEL_PATH = "pt_data/result8.pt"
MODEL_PATH = "/home/juyeong/subway_robot_ws/src/emergency_camera/emergency_camera/subway_project/train_result4/weights/best.pt"
CAM1_IDX = 3
CAM2_IDX = 4

# [설정] 해상도 (MJPG 덕분에 1280도 빠름)
TARGET_W = 1280
TARGET_H = 720
INFERENCE_SIZE = 1280 

# =========================================================
# 1. 모델 로드
# =========================================================
print(f"[{MODEL_PATH}] 모델 로딩 중...")
model = YOLO(MODEL_PATH)

# =========================================================
# 2. 카메라 초기화 (MJPG 활성화)
# =========================================================
def init_camera(idx, width, height):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    
    # [핵심] MJPG 코덱 사용 -> USB 대역폭 확보 -> FPS 급상승
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    return cap

# 카메라 연결 리스트 구성
print(f"카메라 연결 시도 (MJPG Mode, {TARGET_W}x{TARGET_H})...")

active_caps = [] # 활성화된 카메라 정보를 담을 리스트

# 카메라 1 (2번) 연결 시도
cap1 = init_camera(CAM1_IDX, TARGET_W, TARGET_H)
if cap1 is not None and cap1.isOpened():
    print(f"✅ Camera {CAM1_IDX} 연결 성공")
    active_caps.append({'cap': cap1, 'name': f"Cam {CAM1_IDX}"})
else:
    print(f"⚠️ Camera {CAM1_IDX} 연결 실패 (무시하고 진행)")

# 카메라 2 (4번) 연결 시도
cap2 = init_camera(CAM2_IDX, TARGET_W, TARGET_H)
if cap2 is not None and cap2.isOpened():
    print(f"✅ Camera {CAM2_IDX} 연결 성공")
    active_caps.append({'cap': cap2, 'name': f"Cam {CAM2_IDX}"})
else:
    print(f"⚠️ Camera {CAM2_IDX} 연결 실패 (무시하고 진행)")

# 카메라가 하나도 없으면 종료
if not active_caps:
    print("❌ 연결된 카메라가 없습니다. 프로그램을 종료합니다.")
    exit()

print(f"🚀 총 {len(active_caps)}대 카메라로 추론 시작! (Inference Size: {INFERENCE_SIZE})")

# =========================================================
# 3. 실시간 루프
# =========================================================
prev_time = 0

while True:
    frames = []
    valid_caps_info = [] # 이번 턴에 프레임을 성공적으로 읽은 카메라

    # 활성화된 모든 카메라에서 프레임 읽기
    for item in active_caps:
        ret, frame = item['cap'].read()
        if ret:
            frames.append(frame)
            valid_caps_info.append(item)
        else:
            # 일시적인 프레임 드랍은 무시하거나 로그 출력
            pass

    if not frames:
        print("모든 카메라로부터 프레임 수신 실패...")
        break

    # -----------------------------------------------------
    # [추론] MJPG로 압축 전송된 이미지를 OpenCV가 풀어서 추론
    # half=True: 16비트 가속 (화질 영향 없음, 속도 향상)
    # -----------------------------------------------------
    results = model(frames, imgsz=INFERENCE_SIZE, verbose=False, half=True)

    processed_frames = []

    # FPS 계산
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # 결과 그리기 및 정보 표시
    for i, res in enumerate(results):
        res_plot = res.plot()
        cam_name = valid_caps_info[i]['name']
        
        cv2.putText(res_plot, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(res_plot, f"{cam_name} (MJPG)", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        processed_frames.append(res_plot)

    # -----------------------------------------------------
    # [화면 출력] 카메라 개수에 따라 유동적으로 처리
    # -----------------------------------------------------
    if len(processed_frames) == 1:
        # 1대일 경우: 리사이징 없이 원본(또는 적절한 크기) 출력
        final_view = cv2.resize(processed_frames[0], (1280, 720))
        cv2.imshow("Single Cam (MJPG)", final_view)
        
    elif len(processed_frames) >= 2:
        # 2대 이상일 경우: 가로 병합 후 디스플레이용 리사이즈
        combined = cv2.hconcat(processed_frames)
        final_view = cv2.resize(combined, (1920, 540))
        cv2.imshow("Dual Cam (MJPG)", final_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 정리
for item in active_caps:
    item['cap'].release()
cv2.destroyAllWindows()
print("프로그램 종료")