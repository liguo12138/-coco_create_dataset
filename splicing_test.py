"""
task1:需要添加随机数，让objecter随机的出现在前景上
task2:mask的大小应该是可以进行resize的
task3:尽量避免前景与后景objecter之间有太多的相交
task4:需要插入操作链的处理

"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread
from scipy.fftpack import ifftn, fft2, ifft2#傅里叶变换包
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmath
import warnings
import random
# %matplotlib inline
warnings.filterwarnings(action='ignore')


random_h=random.randint(30,100)
random_w=random.randint(40,100)
random_sizeh=random.randint(80,120)
random_sizew=random.randint(70,150)
size=(random_sizew,random_sizeh)

print((random_h,random_w))
print(size)


imgBuilding=np.array(Image.open(r"C:\Users\LG\Desktop\coco_test\img\000000000049.jpg"))#将jpg图片转化为array数组的形式
imgBuildingMask=np.array(Image.open(r"C:\Users\LG\Desktop\coco_test\mask\000000000049.jpg"))
imgBill=np.array(Image.open(r"C:\Users\LG\Desktop\coco_test\img\000000000036.jpg").resize(size))
imgBillMask=np.array(Image.open(r"C:\Users\LG\Desktop\coco_test\mask\000000000036.jpg").resize(size))#这里的mask也是三通道的mask

#imgBill=imgBill[0:imgBill.shape[0]-1]#将图像的高减小了1

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.imshow(imgBuilding)
plt.title("Background")
plt.subplot(2,2,2)
plt.imshow(imgBuildingMask)
plt.subplot(2,2,3)
plt.imshow(imgBill)
plt.title("BillGates")
plt.subplot(2,2,4)
plt.imshow(imgBillMask)
plt.show()

layer0 = imgBuilding.copy()
layer0[imgBuildingMask>128]=0
layer1 = np.zeros(imgBuilding.shape, imgBuilding.dtype)
layer1Mask=np.zeros(imgBuildingMask.shape, imgBuildingMask.dtype)
imgBill[imgBillMask<128]=0#将img_mask上面小于128像素对应的地方全部设为0，等于抠图扣出了了object
layer1[random_h:imgBill.shape[0]+random_h, random_w:imgBill.shape[1]+random_w]=imgBill[:,:]#确定objecter在图像上的位置




layer1Mask[random_h:imgBill.shape[0]+random_h, random_w:imgBill.shape[1]+random_w]=imgBillMask[:,:]
layer2 = imgBuilding.copy()
layer2[imgBuildingMask<128]=0

plt.figure(figsize=(20,10))
plt.subplot(1,3,1)
plt.imshow(layer0)
plt.title("Layer0")
plt.subplot(1,3,2)
plt.imshow(layer1)
plt.title("Layer1")
plt.subplot(1,3,3)
plt.imshow(layer2)
plt.title("Layer2")
plt.show()


layer0[imgBuildingMask>128]=layer2[imgBuildingMask>128]
layer0[layer1Mask>128]=layer1[layer1Mask>128]
plt.imshow(layer0)
plt.show()