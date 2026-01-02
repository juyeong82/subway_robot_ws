import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from datetime import datetime

# 메시지 타입 임포트
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Bool

class TopicMonitorNode(Node):
    def __init__(self):
        super().__init__('topic_monitor_node')

        qos = QoSProfile(depth=10)

        # =========================================================
        # [Subscribers] 토픽 구독 설정
        # =========================================================
        
        # 1. 환자 위치 (YOLO 감지 데이터)
        self.create_subscription(
            PoseStamped, 
            '/target', 
            self.cb_patient_target, 
            qos
        )

        # 2. 웹 UI 클릭 좌표
        self.create_subscription(
            Point, 
            '/control/goal_point', 
            self.cb_ui_click, 
            qos
        )

        # 3. 도착 완료 신호
        self.create_subscription(
            Bool, 
            '/emt_arrival_status', 
            self.cb_arrival_status, 
            qos
        )

        print("\n" + "="*50)
        print(" 📡 Topic Monitor Started (Nav2 Not Required)")
        print("="*50)
        print(" 모니터링 중인 토픽 목록:")
        print("  1. /target (PoseStamped)")
        print("  2. /control/goal_point (Point)")
        print("  3. /emt_arrival_status (Bool)")
        print("-" * 50)
        print(" [대기 중] 신호가 들어오면 아래에 표시됩니다...\n")

    # =====================================================
    # [Callbacks] 데이터 수신 및 출력
    # =====================================================
    
    def get_time_str(self):
        # 현재 시간을 문자열로 반환 (로그용)
        return datetime.now().strftime("%H:%M:%S")

    def cb_patient_target(self, msg: PoseStamped):
        # PoseStamped 메시지 포맷팅 출력
        print(f"\n[{self.get_time_str()}] 🚑 [환자 감지] /target 수신")
        print(f"  ├── Frame ID : {msg.header.frame_id}")
        print(f"  ├── Position : (x: {msg.pose.position.x:.2f}, y: {msg.pose.position.y:.2f})")
        print(f"  └── Orient   : (z: {msg.pose.orientation.z:.2f}, w: {msg.pose.orientation.w:.2f})")
        print("-" * 40)

    def cb_ui_click(self, msg: Point):
        # Point 메시지 포맷팅 출력
        print(f"\n[{self.get_time_str()}] 🖱️ [UI 클릭] /control/goal_point 수신")
        print(f"  ├── Position : (x: {msg.x:.2f}, y: {msg.y:.2f})")
        print(f"  └── Note     : Z값({msg.z})은 보통 무시됨")
        print("-" * 40)

    def cb_arrival_status(self, msg: Bool):
        # Bool 메시지 포맷팅 출력
        status_icon = "✅" if msg.data else "❌"
        status_text = "ARRIVED (접촉 확인)" if msg.data else "NOT ARRIVED"
        
        print(f"\n[{self.get_time_str()}] 🚨 [상태 알림] /emt_arrival_status 수신")
        print(f"  └── Status   : {status_icon} {status_text} ({msg.data})")
        print("-" * 40)

# =========================================================
# [Main] 실행부
# =========================================================
def main(args=None):
    rclpy.init(args=args)
    
    node = TopicMonitorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[종료] 모니터링을 중단합니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()