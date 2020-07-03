import torch
from torch.utils.data import Dataset
from PIL import Image



# 以torch.utils.data.Dataset为基类创建MyDataset
class MyDataset(Dataset):
    # stpe1:初始化
    def __init__(self, txt, transform=None, target_transform=None, ):
        self.root = '/home/yaoting/composition/KU_PCPdataset'
        fh = open(txt, 'r')  # 打开标签文件
        imgs = []
        for line in fh:  # 遍历标签文件每行
            line = line.rstrip()  # 删除字符串末尾的空格
            words = line.split()  # 通过空格分割字符串，变成列表
            imgs.append((words[0], torch.tensor(list(map(int, list(words[1]))))))  # 把图片名words[0]，标签int(words[1])放到imgs里
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 检索函数
        fn, label = self.imgs[index]  # 读取文件名、标签
        img = Image.open(self.root + fn).convert('RGB')  # 通过PIL.Image读取图片
        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     label = torch.take(label, torch.tensor([2, 7, 0, 1, 5, 3, 6, 4, 8]))  # 调整标签顺序
        return img, label

    def __len__(self):
        return len(self.imgs)




