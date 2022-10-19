# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

import numpy as np
from PIL import Image, ImageOps, ImageEnhance


# ImageNet code should change this value
# IMAGE_SIZE = 256


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):  # 随机采样
    return np.random.uniform(low=0.1, high=n)


"""
random.uniform 功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
"""


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


"""
autocontrast是计算输入图像的直方图，去除直方图中最暗和最亮各自cutoff%的直方图值，
找到剩下直方图中最暗和最亮的像素值，以此为单位进行剩余像素值的重新映射，同时使最暗像素为0，
最亮像素为255，即将剩下的像素值再映射到[0,255]的颜色空间上
"""


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


"""
均衡图像的直方图。该函数使用一个非线性映射到输入图像，为了产生灰色值均匀分布的输出图像。
"""


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


"""
将每个颜色通道上变量bits对应的低(8-bits)个bit置0。变量bits的取值范围为[0，8]。
"""


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)  # 双线性插值的方式


def solarize(pil_img, level):  # 在指定的阈值范围内，反转所有的像素点，即原来的值为x，新的像素点的值为255-x。
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)  # 在256-level之上的像素值都需要进行像素值的反转


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((pil_img.width, pil_img.height),
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR)  # (x , y)——>(x+levle*y , y)


def shear_y(pil_img, level, target=False):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:  # 都有类似的这种操作，当随机值选的太大，就反向
        level = -level
    return pil_img.transform((pil_img.width, pil_img.height),
                             Image.AFFINE, (1, 0, 0, level, 1, 0),  # AFFINE（仿射变换）这里（x，y）——>(x,level*x+y)
                             resample=Image.BILINEAR)


"""
(pil_img.width, pil_img.height)指定图像经过仿射变换后和原来的图像是一致的大小
(1, 0, 0, level, 1, 0)为变量data，变量data是一个6元组(a,b,c,d,e,f)，包含一个仿射变换矩阵的第一个两行。
输出图像中的每一个像素（x，y），新值由输入图像的位置（ax+by+c, dx+ey+f）的像素产生，
使用最接近的像素进行近似。这个方法用于原始图像的缩放、转换、旋转和裁剪。这里（x，y）——>(x,level*x+y)
"""


def translate_x(pil_img, level, target=False):  # 水平平移操作
    level = int_parameter(sample_level(level), pil_img.width / 3)  # 对水平平移的最高限度进行一定的限制
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((pil_img.width, pil_img.height),
                             Image.AFFINE, (1, 0, level, 0, 1, 0),  # (x,y)——>(x+level,y)
                             resample=Image.BILINEAR)


def translate_y(pil_img, level, target=False):  # 纵轴上的平移
    level = int_parameter(sample_level(level), pil_img.height / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((pil_img.width, pil_img.height),
                             Image.AFFINE, (1, 0, 0, 0, 1, level),  # （x,y）——>(x,y+level)
                             resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):  # 增强色度
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# python中PIL模块中有一个叫做ImageEnhance的类，该类专门用于图像的增强处理，不仅可以增强（或减弱）图像的亮度、对比度、色度，还可以用于增强图像的锐度。


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level, target=False):  # 对比度增强
    if not target:
        level = float_parameter(sample_level(level), 1.8) + 0.1
        return ImageEnhance.Contrast(pil_img).enhance(level)  # level 表示增强的等级或者说程度
    else:
        return pil_img

# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level, target=False):  # 增强亮度
    if not target:
        level = float_parameter(sample_level(level), 1.8) + 0.1
        return ImageEnhance.Brightness(pil_img).enhance(level)
    else:
        return pil_img


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level,  target=False):  # 锐度
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def zoom_x(pil_img, level):
    level = float_parameter(sample_level(level), 6.0)
    rate = 1.0 / level
    if np.random.random() > 0.5:
        bias = pil_img.width * (1 - rate)
    else:
        bias = 0
    return pil_img.transform((pil_img.width, pil_img.height),
                             Image.AFFINE, (rate, 0, bias, 0, 1, 0),  # （x,y）——>(rate*x+bias,y)
                             resample=Image.BILINEAR)


def zoom_y(pil_img, level):
    level = float_parameter(sample_level(level), 6.0)
    rate = 1.0 / level
    if np.random.random() > 0.5:
        bias = pil_img.height * (1 - rate)
    else:
        bias = 0
    return pil_img.transform((pil_img.width, pil_img.height),
                             Image.AFFINE, (1, 0, 0, 0, rate, bias),  # （x,y）——>(x,rate*y+bias)
                             resample=Image.BILINEAR)#mask的插值应该将插值的方式改成nearest


def convert_top_bottom(pil_img,a):#上下翻折

    #level = float_parameter(sample_level(level), 1.8)  # level的值应该更低
    return pil_img.transpose(Image.FLIP_TOP_BOTTOM)

def mirror(pil_img,a): #镜像
    return pil_img.transpose(Image.FLIP_LEFT_RIGHT)



#
# augmentations = [
#     rotate, shear_x, shear_y,
#     translate_x, translate_y, zoom_x, zoom_y
# ]

#
# augmentations = [
#     autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
#     translate_x, translate_y, convert_top_bottom,mirror
# ]

augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, convert_top_bottom,mirror
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]


# if __name__ == '__main__':
#     img = Image.open(r'C:\Users\LG\Desktop\\test\\adverse\\f10_1_1.jpg')
#     img_mask = Image.open(r'F:\harm_dataset\HFlickr\masks\f10_1.png')
#     # im1 = convert_top_bottom(img)
#     # im1.show()
#
#     np.random.seed(0)#随机种子，保证masks和img有相同的处理方式
#     im2 = zoom_y(img, 3)
#     im2.show()
#
#     np.random.seed(0)
#     im3 = zoom_y(img_mask, 3)
#     im3.show()