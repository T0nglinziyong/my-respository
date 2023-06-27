import torch
import torchmetrics
from matplotlib import pyplot as plt
import PIL
from PIL import Image
import os
from torchmetrics.functional.multimodal.clip_score import *
from torchmetrics.multimodal import CLIPScore
from torchvision import transforms
_ = torch.manual_seed(42)

text1 = "a photo of a cat"
text2 = "a photo of a football"
text3 = 'cat'
text4 = 'football'
text5 = 'ball'
text6 = 'there is a football in the picture with many labels'
text_lst = [text1, text2, text3, text4, text5, text6]
text = 'Two dogs run towards each other on a marshy area.'

metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
trans = [transforms.ToTensor()]  # 通过ToTensor实例将图像数据从PIL类型转换成32位浮点数形式
trans = transforms.Compose(trans)  # 用于串联多个图像变换操作

add_name = 'C:\\Users\\linziyong\\Pictures\\deep learning\\NoCap'
file_lst = os.listdir(add_name)
for file_name in file_lst:
    file = os.path.join(add_name, file_name)
    img = PIL.Image.open(file)
    img = trans(img)
    img = (img * 255)
    img = img.int()
    plt.imshow(img.permute(1, 2, 0))
    print(img.shape)
    plt.show()
    for i, text in enumerate(text_lst):
        score = metric(img, text)
        print(i, score.detach())
        metric.reset()


