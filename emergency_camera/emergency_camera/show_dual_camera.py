import cv2
import os
import datetime

# ==========================================
# [설정 영역]
# ==========================================
CAM1_INDEX = 0           # 첫 번째 카메라 (왼쪽 화면)
CAM2_INDEX = 4           # 두 번째 카메라 (오른쪽 화면)
SAVE_DIR = "train_data"  # 저장할 폴더명
TARGET_W = 1280          # 목표 너비
TARGET_H = 720           # 목표 높이
# ==========================================

# 저장 폴더 생성 확인
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"폴더 확인: {SAVE_DIR}")

def set_camera_props(cap, index):
    """카메라 속성 설정 함수"""
    if not cap.isOpened():
        print(f"Error: {index}번 카메라 연결 실패")
        return False
    
    # 1. MJPG 코덱 설정 (버퍼 지연 방지)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # 2. 해상도 강제 적용
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    return True

def run_dual_collector():
    # 두 카메라 연결
    cap1 = cv2.VideoCapture(CAM1_INDEX)
    cap2 = cv2.VideoCapture(CAM2_INDEX)
    
    # 설정 적용
    if not set_camera_props(cap1, CAM1_INDEX) or not set_camera_props(cap2, CAM2_INDEX):
        return

    print(f"=== 듀얼 데이터 수집 시작 (Cam {CAM1_INDEX} & {CAM2_INDEX}) ===")
    print("[S] 또는 [Space]: 동시 저장")
    print("[Q]: 종료")
    
    count = 0

    while True:
        # 두 카메라 프레임 읽기
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("프레임 수신 실패 (하나 이상의 카메라)")
            break
            
        # 화면 출력용 (두 영상을 가로로 붙임)
        # 만약 해상도가 다르면 resize 필요하지만 현재는 동일 설정이라 바로 붙임
        display = cv2.hconcat([frame1, frame2])
        
        # 정보 표시 (왼쪽 상단에 카운트)
        cv2.putText(display, f"Saved: {count}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # 화면 출력 (창 하나에 두 영상)
        cv2.imshow('Dual Camera Collector', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # 종료
        if key == ord('q'):
            break
            
        # 저장 (S키 또는 스페이스바)
        elif key == ord('s') or key == 32:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 파일명 구분 (cam2_..., cam4_...)
            file1 = f"cam{CAM1_INDEX}_{timestamp}.jpg"
            file2 = f"cam{CAM2_INDEX}_{timestamp}.jpg"
            
            path1 = os.path.join(SAVE_DIR, file1)
            path2 = os.path.join(SAVE_DIR, file2)
            
            # 원본 프레임 저장
            cv2.imwrite(path1, frame1)
            cv2.imwrite(path2, frame2)
            
            print(f"📸 저장됨: {file1}, {file2}")
            count += 1

    # 자원 해제
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dual_collector()