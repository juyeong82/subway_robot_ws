import cv2
import os
import datetime

# ==========================================
# [설정 영역]
# ==========================================
CAM_INDEX = 0 # 0 또는 2 (찍을 카메라 번호 변경)
SAVE_DIR = "homography_data"  # 저장할 폴더명 (용도가 다르니 이름 변경 추천)
TARGET_W = 1280          # 목표 너비
TARGET_H = 720           # 목표 높이
# ==========================================

# 저장 폴더 없으면 생성
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"폴더 확인: {SAVE_DIR}")

# 전역 변수로 클릭한 좌표 저장 리스트 선언
clicked_points = []

# 마우스 이벤트 콜백 함수
def mouse_handler(event, x, y, flags, param):
    global clicked_points
    
    # 왼쪽 마우스 버튼 클릭 시
    if event == cv2.EVENT_LBUTTONDOWN:
        # 좌표가 4개 미만일 때만 추가 (순서대로 1,2,3,4)
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Point {len(clicked_points)} 찍힘: ({x}, {y})")

def run_clean_collector():
    global clicked_points

    # 카메라 연결
    cap = cv2.VideoCapture(CAM_INDEX)
    
    if not cap.isOpened():
        print(f"Error: {CAM_INDEX}번 카메라 연결 실패")
        return

    # 1. MJPG 코덱 설정
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # 2. 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

    # 윈도우 이름 미리 생성 (마우스 콜백 연결용)
    win_name = 'Homography Collector'
    cv2.namedWindow(win_name)
    
    # 마우스 콜백 함수 연결
    cv2.setMouseCallback(win_name, mouse_handler)

    count = 0
    print(f"=== 호모그래피 데이터 수집 (Camera {CAM_INDEX}) ===")
    print("[마우스 왼쪽]: 좌표 찍기 (순서대로 1->4)")
    print("[R]: 찍은 좌표 초기화(다시 찍기)")
    print("[S] 또는 [Space]: 이미지 + 좌표파일 저장")
    print("[Q]: 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 수신 실패")
            break
            
        # 화면 출력용 복사본 생성 (여기에 그림 그림)
        display = frame.copy()
        
        # 클릭한 좌표들을 화면에 표시
        for i, (px, py) in enumerate(clicked_points):
            # 1. 점 찍기 (빨간색)
            cv2.circle(display, (px, py), 5, (0, 0, 255), -1)
            # 2. 번호 매기기 (노란색, 점 옆에 1, 2, 3, 4 표시)
            cv2.putText(display, str(i + 1), (px + 10, py - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 3. 점들끼리 선으로 연결 (시각적 확인용, 선택사항)
            if i > 0:
                prev_x, prev_y = clicked_points[i-1]
                cv2.line(display, (prev_x, prev_y), (px, py), (0, 255, 0), 2)
            # 4번째 점까지 찍히면 마지막 점과 첫 점 연결해서 사각형 닫아주기
            if i == 3:
                fx, fy = clicked_points[0]
                cv2.line(display, (px, py), (fx, fy), (0, 255, 0), 2)

        # 상태 표시
        status_text = f"Points: {len(clicked_points)}/4 | Saved: {count}"
        cv2.putText(display, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(clicked_points) == 4:
            cv2.putText(display, "READY TO SAVE (Press 'S')", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 화면 출력
        cv2.imshow(win_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # 종료
        if key == ord('q'):
            break
        
        # 좌표 초기화 (실수했을 때 R키)
        elif key == ord('r'):
            clicked_points = []
            print("좌표 리셋됨.")

        # 저장 (S키 또는 스페이스바)
        elif key == ord('s') or key == 32:
            if len(clicked_points) < 4:
                print("❌ 경고: 좌표 4개를 모두 찍어야 저장할 수 있음!")
                continue

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 이미지 저장 (마커가 그려진 display 화면을 저장)
            img_filename = f"cam{CAM_INDEX}_{timestamp}.jpg"
            img_path = os.path.join(SAVE_DIR, img_filename)
            cv2.imwrite(img_path, display)
            
            # 2. 좌표 텍스트 파일 저장
            txt_filename = f"cam{CAM_INDEX}_{timestamp}.txt"
            txt_path = os.path.join(SAVE_DIR, txt_filename)
            
            with open(txt_path, "w") as f:
                f.write(f"Image: {img_filename}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write("-" * 20 + "\n")
                for i, (px, py) in enumerate(clicked_points):
                    f.write(f"Point_{i+1}: {px}, {py}\n")
            
            print(f"📸 저장 완료:\n - 이미지: {img_filename}\n - 좌표: {txt_filename}")
            count += 1
            
            # 저장 후 좌표 초기화 할지 말지 결정 (연속 촬영이면 주석 처리)
            # clicked_points = [] 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_clean_collector()