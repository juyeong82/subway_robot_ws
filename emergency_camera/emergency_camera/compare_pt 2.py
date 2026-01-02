import cv2
import os
import pandas as pd
# [설정] GUI 충돌 방지를 위한 Headless 모드
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# ---------------------------------------------------------
# 설정값 정의
# ---------------------------------------------------------
# 모델 인덱스 리스트
target_indices = [1, 2, 3, 4, 5, 6, 7, 8]

# [변경] 사용할 카메라 인덱스 리스트 (2번, 4번)
target_cameras = [2, 4]

# 파일 경로 패턴
model_path_pattern = "pt_data/result{}.pt"

# 결과 저장 폴더 생성
output_dir = "comparison_result"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. 멀티 웹캠 이미지 캡처
# ---------------------------------------------------------
captured_frames = {} # {카메라ID: 이미지배열} 형태로 저장

print(f"카메라 {target_cameras} 캡처 시작...")

for cam_idx in target_cameras:
    cap = cv2.VideoCapture(cam_idx)
    
    if not cap.isOpened():
        print(f"[Skip] 카메라 {cam_idx}번을 열 수 없음. 연결 확인 필요.")
        continue

    # 카메라 노출 안정화 (잠시 대기 후 캡처)
    for _ in range(10): 
        cap.read()
    
    ret, frame = cap.read()
    cap.release()

    if ret:
        captured_frames[cam_idx] = frame
        # 원본 이미지 저장 (파일명에 카메라 번호 포함)
        save_path = os.path.join(output_dir, f"original_capture_cam{cam_idx}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"  - 카메라 {cam_idx} 캡처 완료: {save_path}")
    else:
        print(f"  - [Fail] 카메라 {cam_idx} 프레임 캡처 실패")

if not captured_frames:
    print("캡처된 이미지가 없습니다. 프로그램을 종료합니다.")
    exit()

# ---------------------------------------------------------
# 2. 모델별 추론 및 데이터 수집 (모델 Load 횟수 최소화)
# ---------------------------------------------------------
results_data = [] 

print("\n모델 성능 및 추론 시간 측정 시작...")

for idx in target_indices:
    model_file = model_path_pattern.format(idx)
    
    if not os.path.exists(model_file):
        print(f"[Skip] 모델 파일 없음: {model_file}")
        continue
        
    print(f"[{model_file}] 로드 중...")
    
    try:
        # 모델은 한 번만 로드하고, 여러 이미지를 추론 (속도 최적화)
        model = YOLO(model_file)
        
        # 캡처해둔 각 카메라 이미지에 대해 추론 수행
        for cam_idx, frame in captured_frames.items():
            
            # 추론 실행
            results = model(frame, verbose=False)
            
            for r in results:
                # 시간 계산
                speed = r.speed
                inference_time = speed['inference']
                
                # 결과 이미지 생성
                im_array = r.plot()
                
                # 이미지에 텍스트 표시 (추론 시간 + 카메라 번호)
                info_text = f"Cam:{cam_idx} | Inf:{inference_time:.1f}ms"
                cv2.putText(im_array, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
                # 결과 저장 (파일명: result_모델번호_카메라번호.jpg)
                save_filename = f"result_model_{idx}_cam{cam_idx}.jpg"
                save_path = os.path.join(output_dir, save_filename)
                cv2.imwrite(save_path, im_array)
                
                # 데이터 수집
                boxes = r.boxes
                if len(boxes) == 0:
                     results_data.append({
                        "Model": f"Result {idx}",
                        "Camera": f"Cam {cam_idx}",  # [추가] 카메라 정보
                        "Class": "None",
                        "Confidence": 0.0,
                        "Inference Time (ms)": inference_time
                    })
                else:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        
                        results_data.append({
                            "Model": f"Result {idx}",
                            "Camera": f"Cam {cam_idx}", # [추가] 카메라 정보
                            "Class": cls_name,
                            "Confidence": conf,
                            "Inference Time (ms)": inference_time
                        })
                
    except Exception as e:
        print(f"Error processing model {idx}: {e}")

# ---------------------------------------------------------
# 3. 결과 시각화
# ---------------------------------------------------------
if not results_data:
    print("데이터가 없습니다.")
    exit()

df = pd.DataFrame(results_data)

# 3-1. 정확도(Confidence) 그래프
# 카메라가 2대이므로, 같은 모델이라도 결과가 많아짐 -> 그래프가 복잡할 수 있어 Camera 정보는 툴팁/데이터로 확인
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")

# x=모델, y=정확도, hue=클래스 (카메라 구분 없이 전체 경향성 파악)
# 만약 카메라별로 쪼개서 보고 싶으면 hue="Camera"로 변경 가능하나, 클래스 정보가 사라짐
ax1 = sns.barplot(data=df[df["Class"] != "None"], x="Model", y="Confidence", hue="Class", palette="viridis", errorbar=None)

plt.title("YOLO Model Accuracy Comparison (Combined Cams)", fontsize=15)
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.2f', padding=3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "graph_accuracy.png"))
plt.close()

# 3-2. 추론 시간(Inference Time) 그래프
# 이미지 내용에 따라 추론 시간이 크게 변하지 않으므로, 모델별 평균으로 시각화하거나 산점도로 표현
df_time = df[["Model", "Camera", "Inference Time (ms)"]].drop_duplicates()

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
# hue를 Camera로 설정하여 카메라별 추론 시간 차이(혹은 USB 대역폭 이슈 등) 확인 가능
ax2 = sns.barplot(data=df_time, x="Model", y="Inference Time (ms)", hue="Camera", palette="rocket")

plt.title("Inference Speed by Model & Camera", fontsize=15)
plt.ylabel("Inference Time (ms)", fontsize=12)

for container in ax2.containers:
    ax2.bar_label(container, fmt='%.1f', padding=3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "graph_inference_time.png"))
plt.close()

print("그래프 저장 완료.")

# ---------------------------------------------------------
# 4. CSV 저장
# ---------------------------------------------------------
csv_save_path = os.path.join(output_dir, "detection_results_dual_cam.csv")
# 보기 좋게 정렬 (모델 -> 카메라 -> 클래스 순)
df.sort_values(by=["Model", "Camera", "Class"], inplace=True)
df.to_csv(csv_save_path, index=False)
print(f"상세 데이터 저장 완료: {csv_save_path}")

print("모든 작업 완료.")