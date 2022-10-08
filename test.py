import cv2
import numpy as np
img = np.zeros((10,10), dtype=np.uint8)
print('请看下面图像数据长的样子吧：/n', img, '/n')   #/n表示回车重启一行
cv2.imshow('one', img)         #图像窗口显示图像
img[0:3,3:6,:]=255               #把img图像上的第0行到第2行，第3列到第5列的像素点切出来，并且给它们赋值为255
print('请看切出来像素的值，是不是从0变为255了：/n', img)
cv2.imshow('two', img)
#cv2.waitKey(20000)          #20秒后就执行下面语句吧，程序别老卡在这条语句上了。
#cv2.destroyAllWindows()     #这么图片窗口都统统消失吧

picture_height,picture_width,_ = img.shape
i = 0
# up = -1
for a in range(picture_height):
    for b in range(picture_width):
        if img[a, b].all() > 0:#三通道第a行，第b列的像素值求和？
            # if up == -1:
            #     up = a
            i = i + 1
r = i / (picture_height * picture_width)
print(r)
