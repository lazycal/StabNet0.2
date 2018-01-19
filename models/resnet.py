from torchvision import models
from torch import nn

def resnet50(**kwargs):
    model = models.resnet50(**kwargs)
    model = nn.Sequential(*list(model.children())[:-2])
    return model