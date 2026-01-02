import cv2
import numpy as np
from ultralytics import YOLO

# 1. 학습된 모델 불러오기
# model_path = '/home/juyeong/subway_robot_ws/src/emergency_camera/emergency_camera/models/result01.pt' 
# model_path = '/home/juyeong/subway_robot_ws/src/emergency_camera/emergency_camera/subway_project/train_result/weights/best.pt' 
model_path = '/home/juyeong/subway_robot_ws/src/emergency_camera/emergency_camera/models/result04.pt' 
model = YOLO(model_path)

# ==========================================
# [설정] 카메라 인덱스 및 해상도
# ==========================================
CAM1_INDEX = 0
CAM2_INDEX = 4

TARGET_W = 1280
TARGET_H = 720

def init_camera(index):
    """카메라 초기화 및 해상도 설정 함수"""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ Error: {index}번 카메라 연결 실패")
        return None
    
    # MJPG 코덱 및 해상도 설정
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    
    # 실제 설정 확인
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Camera {index} 연결됨 ({w}x{h})")
    return cap

# 2. 두 카메라 연결
cap1 = init_camera(CAM1_INDEX)
cap2 = init_camera(CAM2_INDEX)

if cap1 is None or cap2 is None:
    print("❌ 카메라 연결 문제로 종료합니다.")
    if cap1: cap1.release()
    if cap2: cap2.release()
    exit()

# 창 크기 조절 가능하도록 설정 (두 화면 합치면 너무 커질 수 있음)
cv2.namedWindow("Dual Camera YOLO", cv2.WINDOW_NORMAL)

print("🚀 듀얼 카메라 추론 시작. 종료하려면 'q'를 누르세요.")

while True:
    # 3. 프레임 읽기 (두 카메라 동시에)
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("프레임 수신 실패 (어느 한 쪽 카메라가 끊김)")
        break

    # 4. 모델 추론 (Batch Inference)
    # 리스트로 묶어서 보내면 YOLO가 알아서 한 번에 처리함 (속도 이득)
    # stream=True는 결과값을 제너레이터로 반환하므로 여기서는 리스트 처리를 위해 끔
    results = model([frame1, frame2], conf=0.5, verbose=False)

    # 5. 결과 시각화
    # results[0]은 frame1 결과, results[1]은 frame2 결과
    res_frame1 = results[0].plot()
    res_frame2 = results[1].plot()

    # 6. 화면 합치기 (가로로 병합)
    # 두 해상도가 같으므로 hconcat 사용 가능
    combined_frame = cv2.hconcat([res_frame1, res_frame2])

    # 7. 화면 출력
    cv2.imshow("Dual Camera YOLO", combined_frame)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap1.release()
cap2.release()
cv2.destroyAllWindows()