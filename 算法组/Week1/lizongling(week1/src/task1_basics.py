import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#读取并显示图片
img = cv.imread('../data/3.jpg')
cv.imshow('data/3.jpg',img)
cv.waitKey(0)#等待按键

#查看尺寸、通道数
print(img.shape)#（高度、宽度、通道数）(573, 860, 3)
print(img.shape[0])#高度，即行数
print(img.shape[1])#宽度，即列数
print(img.shape[2])#通道数（彩图通道数为3，BGR，灰度图没有通道数）

#图像的缩放
#绝对尺寸
img=cv.imread('../data/3.jpg')
rows,cols=img.shape[:2]#rows行 cols列,[:2]是因为彩色图的shape有高度、宽度、通道数，只取前两个（高、宽），灰度图不需要，但此写法可兼容
res=cv.resize(img,(int(0.3*cols),int(0.3*rows)))#看尺寸是先高（行）后宽（列），改尺寸是先宽后高..像素坐标必须是整数，所以用int强转一下。
cv.imshow('resize1',res)#imshow必须包含窗口名（随便写，显示在标题栏）、图片数据
cv.waitKey(0)
#相对尺寸
res1=cv.resize(img,None,fx=0.5,fy=0.5,interpolation=cv.INTER_AREA)#原图，dsize(绝对尺寸)不需要，用None占位，fx,fy,插值方法（减少像素丢失）
cv.imshow('resize2',res1)
cv.waitKey(0)

#图片指定区域裁剪
crop_img1=img[int(rows/2):rows,int(cols/2):cols]#裁剪后=原图[起始行Y：结束行Y,起始列X:结束列X]。这里相当于图片右下区域
cv.imshow('crop_img1',crop_img1)
cv.waitKey(0)
crop_img2=img[0:100,0:100]#左上角100*100的区域
cv.imshow('crop_img2',crop_img2)
cv.waitKey(0)
crop_img3=img[rows-400:rows,cols-400:cols]#相当与173-573，460-860这块区域
cv.imshow('crop_img3',crop_img3)
cv.waitKey(0)
cv.destroyAllWindows()

#灰度转换
img2=cv.imread("../data/2.jpg")
gray=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)#将BGR彩图变成GRAY灰度图
print(gray.shape)#没有了通道数
cv.imshow("img2",img2)
cv.imshow('gray',gray)
cv.waitKey(0)
cv.destroyAllWindows()

#保存文件
save="outputs/task1/gray.jpg"
cv.imwrite(save,gray)
#重新显示
read_img=cv.imread(save)
cv.imshow("read_img",read_img)
cv.waitKey(0)
cv.destroyAllWindows()