import torchvision.models as models
import models_lpf.mymodel as model
import torch

# inception = model.alexrotatenet(use_stn=True)
# print(inception.parameters())
pre_state_dict = torch.load('../weights/stn_vggbn_0001_aug_fre/checkpoint_000.pth.tar')['best_acc1']
print(pre_state_dict)
