import albumentations as A
import cv2

def augment(aug, image):
    augmented = aug(image=image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)
    return augmented['image']

class NoneAug():
    def __call__(self, imgs):
        return imgs

class Hflip():
    def __call__(self, imgs):
        hflip = A.HorizontalFlip(p=1)
        return augment(hflip, imgs)

class Rotate():
    def __call__(self, imgs):
        rotate = A.Rotate(limit=10, p=1)
        return augment(rotate, imgs)

class RBrightnessContrast():
    def __call__(self, imgs):
        rbc = A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2, p=1)
        return augment(rbc, imgs)
    
class GBlur():
    def __call__(self, imgs):
        gblur = A.GaussianBlur(p=1,blur_limit=7)
        return augment(gblur, imgs)
    
class RScale():
    def __call__(self, imgs):
        rscale =  A.RandomScale(p=1,scale_limit=0.5)
        return augment(rscale, imgs)