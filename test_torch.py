import torch

from mydata import MyDataset
import torchvision.transforms as transforms

import random
import PIL.Image as Image

# 调用myDataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
trainset = MyDataset('/home/yaoting/composition/KU_PCPdataset/train_list_multi.txt',
                     transform=transforms.Compose(
                         [transforms.Resize((256, 256)),
                          transforms.ToTensor(),
                          normalize]), target_transform=True)

batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False)
dataiter = iter(train_loader)
images, labels = dataiter.next()
transform1 = transforms.RandomRotation(5, resample=False, expand=True, center=None, fill=None)
transform2 = transforms.RandomRotation(5, resample=False, expand=False, center=None, fill=None)
print(transform1(Image.open('./0230.jpg')))
print(transform1(Image.open('./0230.jpg')))

# images = images.squeeze(0)

print(images.size())
print(images[:, 1, :, :])
print(labels)
print(type(labels))
print(labels.shape)
print(labels[0])
# 标签的选择
new_label = []
for i in range(batch_size):
    nonzero = torch.nonzero(torch.where(labels[i] == 1, torch.tensor(1), torch.tensor(0)))
    lable = nonzero.tolist()
    lable = random.choice(lable)
    new_label.append(lable)
print(torch.tensor(new_label))
labels = torch.tensor(new_label)
print(labels.size())

to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()
print(images)
img = to_img(images.squeeze(0).cpu())
input = to_tensor(img)
# input = transforms.Normalize(mean=mean, std=std)(input).unsqueeze(0).cuda()
print(input)


