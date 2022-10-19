import json
iscrowd=[]
json_path = "F:\dataset\COCO\\annotations\instances_val2017.json"
json_labels = json.load(open(json_path, "r"))

anon=json_labels['annotations']
for anon_info in anon[:1000]:#json_labels['annotations']是一个巨大的由字典组成的列表
    print(anon_info)#得到一个图片的标注信息包括分割和bbox等
    j=anon_info.get('iscrowd')
    iscrowd.append(j)
print(iscrowd)


def Sum(x):#计算简单的嵌套得数值的和
    sums = 0
    for ch in x:
        if isinstance(ch, int):
            sums += ch
        if isinstance(ch, list):
            sums += Sum(ch)#列表中嵌套层次不限2层,要用递归
        if isinstance(ch, tuple):
            sums += Sum(ch)#列表中嵌套层次不限2层,要用递归
    return sums
print(Sum(iscrowd))

