import torch
import torchmetrics
from torchmetrics.functional.multimodal.clip_score import *
from torchmetrics.multimodal import CLIPScore
import torchvision
from torchvision import transforms
import PIL
import os
from visdom import Visdom
from torch.utils.data import Dataset, DataLoader
import json
import cv2
from matplotlib import pyplot as plt


address = "C:\\Users\\linziyong\\Pictures\\deep learning\\NoCap.txt"

with open(file=address, mode='r', encoding='utf-8') as file:
    js = file.read()
    dic = json.loads(js)
licenses = dic['licenses']
info = dic['info']
images = dic['images']
annotations = dic['annotations']


class NoCapDataset(Dataset):
    def __init__(self, images, annotations):
        img_dic = {}
        for image in images:
            id = image['id']
            img_address = image['coco_url']
            img_dic[id] = [img_address]
        for annotation in annotations:
            id = annotation['image_id']
            caption = annotation['caption']
            img_dic[id].append(caption)
        self.data = img_dic

    # 根据索引获取data和label
    def __getitem__(self, index):
        return self.data[index]

    # 获取数据集的大小
    def __len__(self):
        return len(self.data)

metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
trans = [transforms.ToTensor()]  # 通过ToTensor实例将图像数据从PIL类型转换成32位浮点数形式
trans = transforms.Compose(trans)

if __name__ == "__main__":
    dataset = NoCapDataset(images, annotations)
    for i in range(4):
        info = dataset[i]
        img = info[0]
        ann = info[1::]
        cap = cv2.VideoCapture(img)
        if (cap.isOpened()):
            ret, img = cap.read()
            img = trans(img)
            img = img * 255
            img = img.int()
            for caption in ann:
                score = metric(img, caption)
                print(caption, score.detach())
            plt.imshow(img.permute(1, 2, 0))
            plt.show()


