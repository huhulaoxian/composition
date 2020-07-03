from PIL import Image
from torchvision import transforms as tfs
import matplotlib.pyplot as plt
import random
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models


lables = [[0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 0]]
print(lables)
lables = torch.tensor(lables)
print(lables)
new_label = []
for i in range(2):
    nonzero = torch.nonzero(torch.where(lables[i] == 1, torch.tensor(1), torch.tensor(0)))
    lable = nonzero.tolist()
    lable = random.choice(lable)
    new_label.append(lable)
print(torch.tensor(new_label))
img = Image.open('./0007.jpg')
print(img.getbands())
loader = transforms.ToTensor()
img = loader(img)
print(img.size())
print(img.shape)
unloader = transforms.ToPILImage()
img = unloader(img)
print(img.size)
print(img.getbands())
img = np.array(img)
print(img.shape)
model = models.alexnet(pretrained=True)
print(model)

img = Image.open('./0029.jpg')
img = img.resize((255, 255))
print(img.size)
img.save('./0029_resize.jpg')
# 先打补丁
img = transforms.functional.pad(img, 20, padding_mode='symmetric')
img.save('./0029_pad_sym.jpg')
print(img.size)
# 旋转
img = transforms.functional.rotate(img, 8, fill=0)
print(img.size)
img.save('./0029_rotate.jpg')
# 中心裁剪
img = transforms.functional.center_crop(img,255)
img.save('./0029_crop.jpg')


lst = [1, 2, 4, 8, 16, 32]
print(sum(map(lambda x: x ** 2, lst[:3])))









