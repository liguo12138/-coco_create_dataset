"""
	需要修改的地方:
		dataDir,savepath改为自己的路径
		class_names改为自己需要的类
		dataset_list改为自己的数据集名称
"""
from pycocotools.coco import COCO
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

'''
路径参数
'''
# 原coco数据集的路径
dataDir = "F:/dataset/COCO//newdata"
# 用于保存新生成的mask数据的路径
savepath = "F:/dataset/COCO/coco_mask"

'''
数据集参数
'''
# coco有80类，这里写要进行二值化的类的名字
# 其他没写的会被当做背景变成黑色
classes_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'traffic', 'fire hydrant', 'stop sign']

datasets_list = ['train2017']


# 生成保存路径，函数抄的(›´ω`‹ )
# if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


# 生成mask图
def mask_generator(coco, width, height, anns_list):
    mask_pic = np.zeros((height, width))
    # 生成mask - 此处生成的是4通道的mask图,如果使用要改成三通道,可以将下面的注释解除,或者在使用图片时搜相关的程序改为三通道
    for single in anns_list:
        mask_single = coco.annToMask(single)
        mask_pic += mask_single
    # 转化为255
    for row in range(height):
        for col in range(width):
            if (mask_pic[row][col] > 0):
                mask_pic[row][col] = 255
    mask_pic = mask_pic.astype(int)
    #return mask_pic

    #转为三通道
    imgs = np.zeros(shape=(height, width, 3), dtype=np.float32)
    imgs[:, :, 0] = mask_pic[:, :]
    imgs[:, :, 1] = mask_pic[:, :]
    imgs[:, :, 2] = mask_pic[:, :]
    imgs = imgs.astype(np.uint8)
    return imgs


# 处理json数据并保存二值mask
def get_mask_data(annFile, mask_to_save):
    # 获取COCO_json的数据
    coco = COCO(annFile)
    # 拿到所有需要的图片数据的id - 我需要的类别的categories的id是多少
    classes_ids = coco.getCatIds(catNms=classes_names)
    # 取所有类别的并集的所有图片id
    # 如果想要交集，不需要循环，直接把所有类别作为参数输入，即可得到所有类别都包含的图片
    imgIds_list = []
    # 循环取出每个类别id对应的有哪些图片并获取图片的id号
    for idx in classes_ids:
        imgidx = coco.getImgIds(catIds=idx)  # 将该类别的所有图片id好放入到一个列表中
        imgIds_list += imgidx
        print("搜索id... ", imgidx)
    # 去除重复的图片
    imgIds_list = list(set(imgIds_list))  # 把多种类别对应的相同图片id合并

    # 一次性获取所有图像的信息
    image_info_list = coco.loadImgs(imgIds_list)

    # 对每张图片生成一个mask
    for imageinfo in image_info_list:
        # 获取对应类别的分割信息
        annIds = coco.getAnnIds(imgIds=imageinfo['id'], catIds=classes_ids, iscrowd=None)
        anns_list = coco.loadAnns(annIds)
        # 生成二值mask图
        mask_image = mask_generator(coco, imageinfo['width'], imageinfo['height'], anns_list)
        # 保存图片
        file_name = mask_to_save + '/' + imageinfo['file_name'][:-4] + '.jpg'
        plt.imsave(file_name, mask_image)
        print("已保存mask图片: ", file_name)


if __name__ == '__main__':
    # 按单个数据集进行处理
    for dataset in datasets_list:
    # 用来保存最后生成的mask图像目录
        mask_to_save = savepath + '/mask_images/' + dataset  # 三通道mask图存储路径
        #mask_to_save = savepath + '/mask_images_RGBA/' + dataset#四通道mask保存路径
        mkr(savepath + '/mask_images/')
        # 生成路径
        mkr(mask_to_save)
        # 获取要处理的json文件路径
        annFile = '{}/annotations/instances_{}_sub.json'.format(dataDir, dataset)
        # 处理数据
        get_mask_data(annFile, mask_to_save)
        print('Got all the masks of {} from {}'.format(classes_names, dataset))
# ————————————————
# 版权声明：本文为CSDN博主「可乐和可杯」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/qq_44943603/article/details/107699207