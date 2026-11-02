import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#读取图片
img1 = cv.imread('../data/football2.jpg')
print(img1.shape)
#将BGR转化为HSV颜色空间
hsv=cv.cvtColor(img1,cv.COLOR_BGR2HSV)
#生成掩膜
lower_white =np.array([0,0,10])
upper_white =np.array([180,30,255])
mask1=cv.inRange(hsv,lower_white,upper_white)
cv.imshow("mask1",mask1)
cv.waitKey(0)
cv.imwrite("../outputs/mask2.jpg", mask1)
#形态学操作
kernal1=np.ones((9,9),np.uint8)
open1=cv.morphologyEx(mask1,cv.MORPH_CLOSE,kernal1)
contour,hierarchy=cv.findContours(open1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
draw_img1=img1.copy()
cv.drawContours(draw_img1,contour,-1,(0,255,0),2)
cv.imshow("draw_img1",draw_img1)
cv.waitKey(0)
#查找轮廓
detected=False
for cnt1 in contour:
    area = cv.contourArea(cnt1)
    if area > 3000 and area < 5000:
        (circle_x, circle_y), radius = cv.minEnclosingCircle(cnt1)
        # 计算圆心和半径
        center = (int(circle_x), int(circle_y))
        radius = int(radius)
        # 在原图上画外接圆
        cv.circle(draw_img1, center, radius, (255, 0, 0), 2)
        # 在原图上画中心点
        cv.circle(draw_img1, center, 5, (0, 0, 255), -1)
        # 准备在图像上写字（参数分别是：图像，文字，坐标，字体，字号，颜色BGR，粗细）
        text = f"u={center[0]},v={center[1]}"
        cv.putText(draw_img1, text, (center[0] - 50, center[1] - radius), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        print(f"中心坐标：（{center[0]},{center[1]}），面积：{area:.2f}")
        detected = True

if not detected:
    print("未检测到目标")
    cv.putText(draw_img1,"未检测到目标",(20,50),cv.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0),3)
cv.imshow("result1", draw_img1)
cv.imwrite("../outputs/result2.jpg", draw_img1)
cv.waitKey(0)
cv.destroyAllWindows()