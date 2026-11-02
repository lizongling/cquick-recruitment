import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#第一组图片
img = cv.imread('../data/2.jpg')

#将BGR转化为HSV颜色空间（处理颜色，HSV比BGR好用）
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
#生成掩膜
#H（色相）S（饱和度）V（明度）
lower_orange =np.array([11,80,80])#橙色下限
upper_orange =np.array([25,255,255])#橙色上限
#cv.inRange会把在这个范围内的像素变成白色（255），不在的变成黑色（0）
mask1=cv.inRange(hsv,lower_orange,upper_orange)
cv.imshow("img",img)
cv.imshow("mask1",mask1)
cv.waitKey(0)
cv.imwrite("../outputs/task3/mask.jpg", mask1)
#形态学操作
kernal=np.ones((5,5),np.uint8)
open1=cv.morphologyEx(mask1,cv.MORPH_CLOSE,kernal)
cv.imshow("open1",open1)
cv.waitKey(0)
#查找轮廓
#cv.findContours会返回所有轮廓，以及它们的层级关系
#cv.RETR_EXTERNAL表示只寻找最外层轮廓
#cv.CHAIN_APPROX_SIMPLE表示只保存轮廓的拐点，节省内存
contour,hierarchy=cv.findContours(open1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)#左边必须写两样东西，一个是所有的轮廓列表，一个是层级关系。必须放处理好的二值图像
#准备一张用来画图的布（可以复制原图，也可新建一纯黑图）
draw_img=img.copy()
cv.drawContours(draw_img,contour,-1,(0,255,0),2)#-1表示画出所有找到的轮廓,(0,255,0)是BGR，2是线条粗细
cv.imshow("draw_img",draw_img)
cv.waitKey(0)
#在遍历所有轮廓之前，先建一个空的列表
all_valid_cnts=[]
#遍历所有轮廓，过滤太小的噪点
for cnt in contour:#（把列表里的轮廓一个一个拿出来，当前拿出来的记为cnt,contour的缩写）
    area=cv.contourArea(cnt)#计算轮廓面积
    if area>100:#过滤了像素面积小于100的噪点
        all_valid_cnts.append(cnt)
if all_valid_cnts:
        x,y,w,h=cv.boundingRect(np.vstack(all_valid_cnts))#计算外接矩形（x,y是左上角坐标，w,h是宽和高）
        #计算中心坐标
        center_x=x+w//2
        center_y=y+h//2
        #在原图上画外接矩形（蓝色，线宽2）
        cv.rectangle(draw_img,(x,y),(x+w,y+h),(255,0,0),2)
        #在原图上画中心点（红色实心圆）
        cv.circle(draw_img,(center_x,center_y),5,(0,0,255),-1)
        print(f"中心坐标：（{center_x},{center_y}），面积：{area:.2f}")#f：格式化字符串
        cv.imshow("result",draw_img)
        cv.imwrite("../outputs/task3/result.jpg", draw_img)
        cv.waitKey(0)
cv.destroyAllWindows()

#第二组图片
img2 = cv.imread('../data/3.jpg')
hsv=cv.cvtColor(img2,cv.COLOR_BGR2HSV)
lower_orange =np.array([5,80,80])
upper_orange =np.array([20,255,255])
mask2=cv.inRange(hsv,lower_orange,upper_orange)
cv.imshow("mask2",mask2)
cv.waitKey(0)
cv.imwrite("../outputs/task3/mask2.jpg", mask2)
kernal2=np.ones((40,40),np.uint8)
open2=cv.morphologyEx(mask2,cv.MORPH_CLOSE,kernal2)
contour,hierarchy=cv.findContours(open2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
draw_img2=img2.copy()
cv.drawContours(draw_img2,contour,-1,(0,255,0),2)#-1表示画出所有找到的轮廓,(0,255,0)是BGR，2是线条粗细
cv.imshow("draw_img2",draw_img2)
cv.waitKey(0)
for cnt2 in contour:
    area = cv.contourArea(cnt2)
    if area > 10000:
        x, y, w, h = cv.boundingRect(cnt2)
        center_x = x + w // 2
        center_y = y + h // 2
        cv.rectangle(draw_img2, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.circle(draw_img2, (center_x, center_y), 5, (0, 0, 255), -1)
        print(f"中心坐标：（{center_x},{center_y}），面积：{area:.2f}")
        cv.imshow("result2", draw_img2)
        cv.imwrite("../outputs/task3/result2.jpg", draw_img2)
        cv.waitKey(0)
cv.destroyAllWindows()

#第一组图片（其他处理）
img3 = cv.imread('../data/2.jpg')
hsv=cv.cvtColor(img3,cv.COLOR_BGR2HSV)
lower_orange =np.array([11,80,80])
upper_orange =np.array([25,255,255])
mask3=cv.inRange(hsv,lower_orange,upper_orange)
cv.imshow("mask3",mask3)
cv.waitKey(0)
cv.imwrite("../outputs/task3/mask3.jpg", mask3)
kernal2=np.ones((40,40),np.uint8)
open3=cv.morphologyEx(mask3,cv.MORPH_CLOSE,kernal2)
contour,hierarchy=cv.findContours(open3,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
draw_img3=img3.copy()
cv.drawContours(draw_img3,contour,-1,(0,255,0),2)#-1表示画出所有找到的轮廓,(0,255,0)是BGR，2是线条粗细
cv.imshow("draw_img3",draw_img3)
cv.waitKey(0)
for cnt3 in contour:
    area = cv.contourArea(cnt3)
    if area > 10000:
        x, y, w, h = cv.boundingRect(cnt3)
        center_x = x + w // 2
        center_y = y + h // 2
        cv.rectangle(draw_img3, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.circle(draw_img3, (center_x, center_y), 5, (0, 0, 255), -1)
        print(f"中心坐标：（{center_x},{center_y}），面积：{area:.2f}")
        cv.imshow("result3", draw_img3)
        cv.imwrite("../outputs/task3/result3.jpg", draw_img3)
        cv.waitKey(0)
cv.destroyAllWindows()

#第三组图片
img4 = cv.imread('../data/4.jpg')
hsv=cv.cvtColor(img4,cv.COLOR_BGR2HSV)
lower_orange =np.array([15,80,80])
upper_orange =np.array([30,255,255])
mask4=cv.inRange(hsv,lower_orange,upper_orange)
cv.imshow("mask4",mask4)
cv.waitKey(0)
cv.imwrite("../outputs/task3/mask4.jpg", mask4)
kernal2=np.ones((40,40),np.uint8)
open4=cv.morphologyEx(mask4,cv.MORPH_CLOSE,kernal2)
contour,hierarchy=cv.findContours(open4,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
draw_img4=img4.copy()
cv.drawContours(draw_img4,contour,-1,(0,255,0),2)#-1表示画出所有找到的轮廓,(0,255,0)是BGR，2是线条粗细
cv.imshow("draw_img4",draw_img4)
cv.waitKey(0)
for cnt4 in contour:
    area = cv.contourArea(cnt4)
    if area > 10000:
        x, y, w, h = cv.boundingRect(cnt4)
        center_x = x + w // 2
        center_y = y + h // 2
        cv.rectangle(draw_img4, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.circle(draw_img4, (center_x, center_y), 5, (0, 0, 255), -1)
        print(f"中心坐标：（{center_x},{center_y}），面积：{area:.2f}")
        cv.imshow("result4", draw_img4)
        cv.imwrite("../outputs/task3/result4.jpg", draw_img4)
        cv.waitKey(0)
cv.destroyAllWindows()

#第四组图片
img5 = cv.imread('../data/5.jpg')
hsv=cv.cvtColor(img5,cv.COLOR_BGR2HSV)
lower_orange =np.array([18,90,75])
upper_orange =np.array([30,255,255])
mask5=cv.inRange(hsv,lower_orange,upper_orange)
cv.imshow("mask5",mask5)
cv.waitKey(0)
cv.imwrite("../outputs/task3/mask5.jpg", mask5)
kernal2=np.ones((30,30),np.uint8)
open5=cv.morphologyEx(mask5,cv.MORPH_CLOSE,kernal2)
contour,hierarchy=cv.findContours(open5,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
draw_img5=img5.copy()
cv.drawContours(draw_img5,contour,-1,(0,255,0),2)#-1表示画出所有找到的轮廓,(0,255,0)是BGR，2是线条粗细
cv.imshow("draw_img5",draw_img5)
cv.waitKey(0)
for cnt5 in contour:
    area = cv.contourArea(cnt5)
    if 50000 <area:
        x, y, w, h = cv.boundingRect(cnt5)
        center_x = x + w // 2
        center_y = y + h // 2
        cv.rectangle(draw_img5, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.circle(draw_img5, (center_x, center_y), 5, (0, 0, 255), -1)
        print(f"中心坐标：（{center_x},{center_y}），面积：{area:.2f}")
        cv.imshow("result5", draw_img5)
        cv.imwrite("../outputs/task3/result5.jpg", draw_img5)
        cv.waitKey(0)
cv.destroyAllWindows()