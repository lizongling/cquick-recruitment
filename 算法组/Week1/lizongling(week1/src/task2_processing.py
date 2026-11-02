import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#灰度图
img=cv.imread("../data/2.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)
cv.waitKey(0)
cv.destroyAllWindows()
img2=cv.imread("../data/4.jpg")


#高斯滤波
#1.图像噪声的类型：椒盐噪声 高斯噪声等
#椒盐噪声：随机出现的白点或黑点，可能是亮的区域有黑色像素或是在暗的区域有白色像素（或是两者都有）
  #成因如失效的感应器导致像素值为最小值，饱和的感应器导致像素值为最大值
#高斯噪声：噪声密度函数服从高斯分布（正态分布）
#2.图像平滑
#去除其中的高频信息，保留低频信息，因此我们可以对图像实施低通滤波。低频滤波可以去除图像中的噪声，对图像进行平滑。
#根据滤波的不同可分为均值滤波，高斯滤波，中值滤波，双边滤波

#高斯平滑（先理解含义）的流程：1，首先确定权重矩阵 2.计算高斯模糊
#cv2.GaussianBlur(src,ksize,sigmaX,borderType)
#src:输入图像。ksize:高斯卷积核的大小，注意:卷积核的宽度和高度都应为奇数，且可以不同。sigmaX:水平方向的标准差。sigmaY:垂直方向的标准差，默认值为0，表示与sigmaX相同。borderType:填充边界类型
cv.imshow("img",img)
cv.waitKey(0)
gs1=cv.GaussianBlur(img,(3,3),0)
cv.imshow("gs1",gs1)
cv.waitKey(0)
gs2=cv.GaussianBlur(img,(5,5),0)
cv.imshow("gs2",gs2)
cv.waitKey(0)
gs3=cv.GaussianBlur(img,(11,11),0)
cv.imshow("gs3",gs3)
cv.waitKey(0)
gs4=cv.GaussianBlur(img,(3,11),0)
cv.imshow("gs4",gs4)
cv.waitKey(0)
gs5=cv.GaussianBlur(img,(11,3),0)
cv.imshow("gs5",gs5)
cv.waitKey(0)
cv.imwrite("../outputs/task2/gs1.jpg", gs1)
cv.imwrite("../outputs/task2/gs2.jpg", gs2)
cv.imwrite("../outputs/task2/gs3.jpg", gs3)
cv.imwrite("../outputs/task2/gs4.jpg", gs4)
cv.imwrite("../outputs/task2/gs5.jpg", gs5)
cv.destroyAllWindows()
gs11=cv.GaussianBlur(img2,(3,3),0)

#阈值分割
ret1,thresh1=cv.threshold(gs1,50,255,cv.THRESH_BINARY)#100是卡的阈值，255是超过分数线后变成什么颜色（255 白色）
cv.imshow("thresh1",thresh1)
cv.waitKey(0)
cv.imwrite("../outputs/task2/thresh1.jpg", thresh1)
ret2,thresh2=cv.threshold(gs1,100,255,cv.THRESH_BINARY)#ret不可省，可用下划线代替。thresh是阈值分割的简写
cv.imshow("thresh2",thresh2)
cv.waitKey(0)
cv.imwrite("../outputs/task2/thresh2.jpg", thresh2)
ret3,thresh3=cv.threshold(gs1,150,255,cv.THRESH_BINARY)
cv.imshow("thresh3",thresh3)
cv.waitKey(0)
#对比：不加高斯滤波直接进行阈值分割
ret4,thresh4=cv.threshold(img,150,255,cv.THRESH_BINARY)
cv.imshow("thresh4",thresh4)
cv.waitKey(0)
cv.destroyAllWindows()

#第二组图片重复
ret11,thresh11=cv.threshold(gs11,50,255,cv.THRESH_BINARY)
cv.imshow("thresh11",thresh11)
cv.waitKey(0)
cv.imwrite("../outputs/task2/thresh1.jpg", thresh1)
ret21,thresh21=cv.threshold(gs11,100,255,cv.THRESH_BINARY)
cv.imshow("thresh21",thresh21)
cv.waitKey(0)
cv.imwrite("../outputs/task2/thresh2.jpg", thresh2)
ret31,thresh31=cv.threshold(gs11,150,255,cv.THRESH_BINARY)
cv.imshow("thresh31",thresh31)
cv.waitKey(0)
cv.destroyAllWindows()

#Canny边缘图
#1.边缘检测原理(相当于提取边缘的图像，类似图像勾线)
#图像边缘检测大幅度地减少了数据量，并且剔除了可以认为不相关的信息，保留了图像重要的结构属性，有很多方法用于边缘检测，它们绝大部分可以分为两类：基于搜索和基于零穿越
#基于搜索：通过寻找图像一阶导数中的最大值来检测边界，然后利用计算结果估计边缘的局部方向，通常采用梯度的方向，并利用此方向找到局部梯度模的最大值，代表算法是Sobel算子和Scharr算子
#基于零穿越：通过寻找图像二阶导数零穿越来寻找边界，代表算法是Laplacian算子
#Canny边缘检测（被认为是最优的边缘检测算法）
#canny=cv2.Canny(image,threshold1,threshold2)
#image:灰度图。threshold1：minVal,较小的阈值将间断的边缘连接起来。threshold2：maxVal,较大的阈值检测图像中明显的边缘
canny1=cv.Canny(gs1,50,150)
cv.imshow("canny",canny1)
cv.waitKey(0)
cv.imwrite("../outputs/task2/canny.jpg", canny1)
canny2=cv.Canny(gs1,50,100)
cv.imshow("canny2",canny2)
cv.waitKey(0)
cv.imwrite("../outputs/task2/canny2.jpg", canny2)
canny3=cv.Canny(gs1,100,150)
cv.imshow("canny3",canny3)
cv.waitKey(0)
cv.imwrite("../outputs/task2/canny3.jpg", canny3)
cv.destroyAllWindows()

#第二组图片重复
canny11=cv.Canny(gs11,50,150)
cv.imshow("canny11",canny11)
cv.waitKey(0)
canny21=cv.Canny(gs11,50,100)
cv.imshow("canny21",canny21)
cv.waitKey(0)
canny31=cv.Canny(gs11,100,150)
cv.imshow("canny31",canny31)
cv.waitKey(0)
cv.destroyAllWindows()


#形态学操作

#创建核结构
kernel=np.ones((5,5),np.uint8)

#腐蚀
#腐蚀的作用是消除物体边界点，使目标缩小，可以消除小于结构元素的噪声点
#cv.erode(img,kernel,iterations)
#参数 img:要处理的图像 kernel:核结构 iterations:腐蚀的次数，默认是1
cv.imshow("gs1",gs1)
cv.waitKey(0)
fushi1=cv.erode(gs1,kernel)
cv.imshow("fushi1",fushi1)
cv.waitKey(0)
cv.imwrite("../outputs/task2/fushi1.jpg", fushi1)

#膨胀
#膨胀的作用是将与物体接触的所有背景点合并到物体中，使目标增大，可添补目标中的孔洞
#cv.dilate(img,kernel,iterations)
#参数 img:要处理的图像 kernel:核结构 iterations:腐蚀的次数，默认是1
pengzhang1=cv.dilate(gs1,kernel)
cv.imshow("pengzhang1",pengzhang1)
cv.waitKey(0)
cv.imwrite("../outputs/task2/pengzhang1.jpg", pengzhang1)

#开运算和闭运算（不可逆）
#开运算：先腐蚀后膨胀。作用是分离物体，消除小区域。特点是消除噪点，去除小的干扰块，而不影响原来的图像
#闭运算：先膨胀后腐蚀。作用是消除/闭合物体里面的孔洞，特点是可以填充闭合区域
#cv.morphologyEx(img,op,kernel)
#参数 op:处理方式，开运算则为从cv.MORPH_OPEN,闭运算则为cv.MORPH_CLOSE

#开运算
kernel=np.ones((5,5),np.uint8)
open_1=cv.morphologyEx(gs1,cv.MORPH_OPEN,kernel)
cv.imshow("open_1",open_1)
cv.waitKey(0)
cv.imwrite("../outputs/task2/open1.jpg", open_1)

#闭运算
kernel=np.ones((5,5),np.uint8)
close_1=cv.morphologyEx(gs1,cv.MORPH_CLOSE,kernel)
cv.imshow("close_1",close_1)
cv.waitKey(0)
cv.imwrite("../outputs/task2/close1.jpg", close_1)
cv.destroyAllWindows()

#第二组照片改变核大小重复
#开运算
kernel2=np.ones((3,3),np.uint8)
open_2=cv.morphologyEx(gs11,cv.MORPH_OPEN,kernel2)
cv.imshow("open_2",open_2)
cv.waitKey(0)
cv.imwrite("../outputs/task2/open2.jpg", open_2)

#闭运算
close_2=cv.morphologyEx(gs11,cv.MORPH_CLOSE,kernel2)
cv.imshow("close_2",close_2)
cv.waitKey(0)
cv.imwrite("../outputs/task2/close2.jpg", close_2)
cv.destroyAllWindows()