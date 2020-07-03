import torch
import numpy as np

# batch : 4
targets = torch.LongTensor([[1, 0, 0, 1, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 1, 0, 0]])
# ktop : 2
predicts = torch.LongTensor([[1, 3],
                             [2, 3],
                             [3, 4],
                             [1, 4]])
print(targets)
print(predicts)
predicts = predicts.t()
print(predicts)

# label = torch.LongTensor([0, 1, 0, 1])
# mul = label.mul(target)
# print(mul)
# print(torch.where(mul>0)[0])
# label to onehot
batch_size = 5
class_num = 9
ktop = 3
predicts = np.random.randint(0, class_num, size=(batch_size, ktop))
predicts = torch.LongTensor(predicts)
print('predicts:', predicts)
onehot = torch.zeros(batch_size, class_num).scatter(1, predicts, 1)
print('onehot:', onehot)
targets = torch.tensor([[0., 0., 0., 0., 0., 1., 0., 1., 1.],
                        [0., 1., 0., 0., 0., 0., 1., 0., 1.],
                        [0., 1., 0., 0., 0., 0., 1., 1., 0.],
                        [0., 0., 0., 1., 0., 1., 0., 0., 1.],
                        [1., 0., 1., 0., 1., 0., 0., 0., 0.]])
print('targets', targets)

correct = onehot.mul(targets)
print('correct:', correct)
# 横向相加 dim = 1 纵向相加 dim = 0
sum = correct.sum(1)
print('sum:', sum)
# 非0元素变为1  torch.where(sum > 0, torch.tensor(1), torch.tensor(0))
correct_k = torch.where(sum > 0, torch.tensor(1), torch.tensor(0))
print(correct_k)

correct_k = torch.where(sum > 0, torch.tensor(1), torch.tensor(0)).float().sum(0, keepdim=True).mul_(100.0 / batch_size)
print(correct_k)
print(correct_k.dtype)
print(correct_k[0])

torch.cuda.set_device(1)
target = torch.tensor([0, 1, 0, 1, 2])
nonzero = torch.nonzero(torch.where(target == 1, torch.tensor(1), torch.tensor(0)))



probs = torch.tensor([[0., 0., 4., 0., 0., 4., 0., 1., 1.],
                      [0., 1., 6., 0., 0., 0., 1., 0., 1.],
                      [0., 1., 0., 5., 0., 0., 1., 1., 0.],
                      [0., 0., 0., 1., 0., 1., 7., 0., 1.],
                      [1., 0., 1., 0., 1., 0., 0., 7., 0.]])
# argmax 返回的是该维度最大值的索引
print(probs.numpy()[:,[0,1]]) #取0，1列
print(probs.argmax(dim=1).cpu().data.numpy())
corrs0 = probs.argmax(dim=1).cpu().data.numpy() == 3
corrs1 = probs.argmax(dim=1).cpu().data.numpy() == 2
print(corrs0)
print(corrs1)
corrs = corrs0 | corrs1
print(corrs)
input = torch.tensor([[[1],[2],[3],[4]]])
print(input.dim())
inputs = []
for i in range(5):
    inputs.append(input)
print(inputs)
inputs = torch.cat(inputs, dim=0)
print(inputs)

rotate_probs = np.zeros((5, 10))
print(rotate_probs)
rotate_probs[0,0] = 4
print(rotate_probs)

probs = torch.tensor([[0., 0., 4., 0., 0., 4., 0., 1., 1.]])
print(torch.nonzero(probs[0])[0].item())
probs = torch.tensor([[0., 0., 4., 0., 0., 4., 0., 1., 1.]])
print(probs)
print(probs.squeeze(0))
print(probs)


out = 1
for i in range(2):
    print(i)
    out = i+out
print(out)
x = torch.tensor([1, 2, 3])
print(x.repeat(4,1))
print(np.arange(-8, 8.5, 0.5).tolist())

theta = torch.tensor([1, 0, 0, 1])
theta = theta.view(2,2)
pad = torch.tensor([[0],[0]])
print(pad)
theta = torch.cat((theta,pad),1)
print(theta)
pad = torch.zeros(5,2,1)
print(pad)

if (0 % 10 == 0):
    print("-----------")
