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
"""Reference implementation of AugMix's data augmentation method in numpy."""
import augmentations
import numpy as np
import random
from PIL import Image

# CIFAR-10 constants
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    '''
    image = image.transpose(2, 0, 1)  # Switch to channel-first
    mean, std = np.array(MEAN), np.array(STD)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image.transpose(1, 2, 0)
    '''
    return image


def apply_op(image, op, severity,target=False):
    image = np.clip(image, 0, 255).astype(np.uint8)#将0-1的图片 放大到0-255
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity, target)
    return np.asarray(pil_img)


def augment_and_mix(image, severity=1, width=3, depth=-1, alpha=1.):  # 在这里执行augmix
    #这里的augmix对于篡改图片可能不适用，考虑用先后的操作将其串联起来而不是用凸组合的方式

    """Perform AugMix augmentations and compute mixture.

    Args:
      image: Raw input image as float32 np.ndarray of shape (h, w, c)
      severity: Severity of underlying augmentation operators (between 1 to 10).
      width: Width of augmentation chain
      depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:
      mixed: Augmented and mixed image.
    """
    ws = np.float32(
        np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image)
    for i in range(width):
        image_aug = image.copy()#复制一个拷贝
        depth = depth if depth > 0 else np.random.randint(2, 4)
        for _ in range(depth):#
            op = np.random.choice(augmentations.augmentations_all)#op表示选择的处理方式，这里是等概率的，如果加入mirror以及翻折可以设置一下概率
            print(op)
            # 这里需不需要将replace设置False呢表示能取一样的值？
            # print(op)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * normalize(image_aug)

    max_ws = max(ws)
    rate = 1.0 / max_ws
    # print(rate)

    # mixed = (random.randint(5000, 9000)/10000) * normalize(image) + (random.randint((int)(rate*3000), (int)(rate*10000))/10000) * mix
    mixed = max((1 - m), 0.7) * normalize(image) + max(m, rate * 0.5) * mix
    # mixed = (1 - m) * normalize(image) + m * mix
    return mixed



def augment_and_list(image, severity=2, count=5,target=False):#

    #mix = np.zeros_like(image)
    for _ in range(count):  #
        image_aug = image#.copy()
        op = np.random.choice(augmentations.augmentations_all)  # op表示选择的处理方式，这里是等概率的，如果加入mirror以及翻折可以设置一下概率
        print(op)
        image_aug = apply_op(image_aug, op, severity, target)
        #mix = normalize(image_aug)
        image = image_aug#.copy()
    return image
"""

"""
if __name__ == '__main__':
    img = Image.open(r'C:\Users\LG\Desktop\\test\\adverse\\f10_1_1.jpg')
    img_mask = Image.open(r'F:\harm_dataset\HFlickr\masks\f10_1.png')#.convert('1')
    img.show()
    img = np.array(img).astype(np.float64)

    img_mask = np.array(img_mask).astype(np.float64)#这里的图片是0和255形式，8位的图
    # im1 = convert_top_bottom(img)
    # im1.show()

    np.random.seed(10)#随机种子，保证masks和img有相同的处理方式
    im2 = augment_and_list(img,target=None)#[0,255]->[0,255]
    im2=Image.fromarray(np.uint8(im2))
    im2.show()

    np.random.seed(10)
    im3 = augment_and_list(img_mask,target=True)
    im3=Image.fromarray(np.uint8(im3))
    im3.show()