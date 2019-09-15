import os
from PIL import Image
import numpy as np
from torchvision import transforms
from .utils import *
import torch.nn.functional as F
import cv2

tta_aug = [
    NoneAug(),
    Hflip(),
    Rotate(),
    RBrightnessContrast(),
    GBlur(),
    RScale(),
]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean, std)
transformList = []
transformList.append(transforms.Resize((480, 480)))
transformList.append(transforms.ToTensor())
transformList.append(normalize)  
transformSequence = transformSequenceVal = transforms.Compose(transformList)

class ClassTTAPredictor():

    def __init__(self, model,device, augs = tta_aug):
        self.model = model
        self.augs = augs
        self.device = device

    def predict(self,inputs):
        self.preds = []
        for aug in self.augs:
            self.preds.append(self._predict_single(inputs,aug))
        self.preds = np.mean(np.array(self.preds),axis=0)

        return self.preds

    def _predict_single(self, imgs, aug):
        aug_imgs = aug(imgs)
        im = Image.fromarray(np.uint8(aug_imgs))
        pred = self.model((transformSequenceVal(im).to(self.device).unsqueeze(0))).data.cpu().numpy()[0]
        return pred

TTA = ClassTTAPredictor(model, device, tta_aug)
image = cv2.imread(image_url)
predTTA = TTA.predict(image)