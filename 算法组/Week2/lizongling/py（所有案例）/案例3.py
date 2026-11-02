import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#读取图片
img1 = cv.imread('../data/football3.jpg')
print(img1.shape)
cv.imshow('img1',img1)
cv.waitKey(0)
#将BGR转化为HSV颜色空间
hsv=cv.cvtColor(img1,cv.COLOR_BGR2HSV)
#生成掩膜
lower_white =np.array([0,15,50])
upper_white =np.array([180,30,255])
mask1=cv.inRange(hsv,lower_white,upper_white)
cv.imshow("mask1",mask1)
cv.waitKey(0)
cv.imwrite("../outputs/mask3.jpg", mask1)
#形态学操作
kernal1=np.ones((33,9),np.uint8)
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
    if area > 3000:
        x, y, w, h = cv.boundingRect(cnt1)
        #计算长宽比
        aspect_ratio = w/float(h)
        if aspect_ratio >0.7 and aspect_ratio <1.4:
            #计算中心坐标
            center_x = x + w // 20
            center_y = y + h // 2
            #在原图上画外接矩形
            cv.rectangle(draw_img1, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #在原图上画中心点
            cv.circle(draw_img1, (center_x, center_y), 5, (0, 0, 255), -1)
            #准备在图像上写字（参数分别是：图像，文字，坐标，字体，字号，颜色BGR，粗细）
            text = f"u={center_x},v={center_y}"
            cv.putText(draw_img1,text,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)

            print(f"中心坐标：（{center_x},{center_y}），面积：{area:.2f}")
            detected = True

if not detected:
    print("未检测到目标")
    cv.putText(draw_img1,"未检测到目标",(20,50),cv.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0),3)
cv.imshow("result1", draw_img1)
cv.imwrite("../outputs/result3.jpg", draw_img1)
cv.waitKey(0)
cv.destroyAllWindows()