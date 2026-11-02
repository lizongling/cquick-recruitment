import rclpy
from rclpy.node import Node
from my_interfaces.msg import TargetPosition  # 导入你写好的消息包
import cv2 as cv
import numpy as np
import os

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_processing_node')
        self.get_logger().info('Vision processing node has been started!')
        
        # 创建发布者，发布到 /target_position 话题
        self.publisher_ = self.create_publisher(TargetPosition, '/target_position', 10)
        
        self.home_dir = os.path.expanduser('~')
        self.image_path = os.path.join(self.home_dir, 'test.jpg')
        
        # 把原来的 process_image 放进定时器里，每秒执行一次（相当于死循环）
        self.timer = self.create_timer(1.0, self.process_image)
        
    def process_image(self):
        img1 = cv.imread(self.image_path)
        if img1 is None:
            self.get_logger().error(f"Failed to read image: {self.image_path}")
            return

        hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
        
        lower_white = np.array([0, 0, 10])
        upper_white = np.array([180, 30, 255])
        mask1 = cv.inRange(hsv, lower_white, upper_white)
        
        kernel1 = np.ones((9, 9), np.uint8)
        open1 = cv.morphologyEx(mask1, cv.MORPH_CLOSE, kernel1)
        
        # 注意你的 OpenCV 版本里 findContours 返回几个值，如果是3个，这里要改
        contour, hierarchy = cv.findContours(open1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        draw_img1 = img1.copy()
        cv.drawContours(draw_img1, contour, -1, (0, 255, 0), 2)
        
        detected = False
        
        # ================= 准备要发送的消息对象 =================
        msg = TargetPosition()
        msg.detected = False
        msg.u = 0.0
        msg.v = 0.0
        msg.area = 0.0
        # ========================================================
        
        for cnt1 in contour:
            area = cv.contourArea(cnt1)
            if area > 3000 and area < 5000:
                (circle_x, circle_y), radius = cv.minEnclosingCircle(cnt1)
                center = (int(circle_x), int(circle_y))
                radius = int(radius)
                
                cv.circle(draw_img1, center, radius, (255, 0, 0), 2)
                cv.circle(draw_img1, center, 5, (0, 0, 255), -1)
                
                text = f"u={center[0]},v={center[1]}"
                cv.putText(draw_img1, text, (center[0] - 50, center[1] - radius), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # 填充真实数据到消息里
                msg.detected = True
                msg.u = float(center[0])
                msg.v = float(center[1])
                msg.area = float(area)
                
                detected = True
                break # 找到一个就退出
                
        if not detected:
            cv.putText(draw_img1, "No Target", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            self.get_logger().info('发送: 未检测到目标')
        else:
            self.get_logger().info(f'发送目标: u={msg.u}, v={msg.v}, area={msg.area}')
            
        # ================= 核心：发布消息 =================
        self.publisher_.publish(msg)
        # ==================================================

        cv.imshow("result1", draw_img1)
        
        # 【修复卡死问题】：必须是 1，不能是 0
        cv.waitKey(1) 

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    
    try:
        # 用 spin 替代直接调用 process_image，让 ROS 2 保持运行和通信
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
