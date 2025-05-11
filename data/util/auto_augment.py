import random
import numpy as np
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps


class AutoAugment(object):
    def __init__(self):
        self.policies = [            
            ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
            ['Contrast', 0.5, 8, 'Equalize', 0.9, 2],
            ['Posterize', 0.3, 7, 'Brightness', 0.6, 7],
            ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
            ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
            ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
            ['Equalize', 0.3, 7, 'Contrast', 0.4, 8],
            ['Brightness', 0.9, 6, 'Contrast', 0.2, 8],
            ['Equalize', 0.2, 0, 'Contrast', 0.6, 0],
            ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
            ['Contrast', 0.8, 4, 'Brightness', 0.2, 8],
            ['Brightness', 0.1, 3, 'Contrast', 0.7, 0],
            ['Contrast', 0.9, 2, 'Brightness', 0.8, 3],
            ['Equalize', 0.8, 8, 'Contrast', 0.1, 3],
            ['Contrast', 0.9, 1, 'Brightness', 0.7, 9],
        ]

    def __call__(self, img):
        img = apply_policy(img, self.policies[random.randrange(len(self.policies))])
        return img


class ImageNetAutoAugment(object):
    def __init__(self):
        self.policies = [
            ['Posterize', 0.4, 8, 'Rotate', 0.6, 9],
            ['Equalize', 0.8, 8, 'Equalize', 0.6, 3],
            ['Posterize', 0.6, 7, 'Posterize', 0.6, 6],
            ['Equalize', 0.4, 7, 'Contrast', 0.2, 4],
            ['Equalize', 0.4, 4, 'Rotate', 0.8, 8],
            ['Posterize', 0.8, 5, 'Equalize', 1.0, 2],
            ['Rotate', 0.2, 3, 'Brightness', 0.6, 8],
            ['Equalize', 0.6, 8, 'Posterize', 0.4, 6],
            ['Rotate', 0.8, 8, 'Contrast', 0.4, 0],
            ['Rotate', 0.4, 9, 'Equalize', 0.6, 2],
            ['Equalize', 0.0, 0.7, 'Equalize', 0.8, 8],
            ['Contrast', 0.6, 4, 'Equalize', 1.0, 8],
            ['Rotate', 0.8, 8, 'Contrast', 1.0, 2],
            ['Sharpness', 0.4, 7, 'Equalize', 0.6, 8],
            ['Equalize', 0.4, 7, 'Contrast', 0.2, 4],
            ['Equalize', 0.8, 8, 'Equalize', 0.6, 3]
        ]

    def __call__(self, img):
        img = apply_policy(img, self.policies[random.randrange(len(self.policies))])
        return img


operations = {
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'Equalize': lambda img, magnitude: equalize(img, magnitude),
    'Posterize': lambda img, magnitude: posterize(img, magnitude),
    'Contrast': lambda img, magnitude: contrast(img, magnitude),
    'Brightness': lambda img, magnitude: brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: cutout(img, magnitude),
}


def apply_policy(img, policy):
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
    if random.random() < policy[4]:
        img = operations[policy[3]](img, policy[5])

    return img


def rotate(img, magnitude):
    img = np.array(img)
    angles = [90, 180, 270, -90, -180, -270]
    angle = random.choice(angles)
    img = ndimage.rotate(img, angle, reshape=False)
    img = Image.fromarray(img)
    return img



def equalize(img, magnitude):
    img = ImageOps.equalize(img)
    return img


def posterize(img, magnitude):
    magnitudes = np.linspace(4, 8, 11)
    img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))))
    return img


def contrast(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def brightness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def sharpness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def cutout(org_img, magnitude=None):
    magnitudes = np.linspace(0, 60/331, 11)

    img = np.copy(org_img)
    mask_val = img.mean()

    if magnitude is None:
        mask_size = 16
    else:
        mask_size = int(round(img.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])))
    top = np.random.randint(0 - mask_size//2, img.shape[0] - mask_size)
    left = np.random.randint(0 - mask_size//2, img.shape[1] - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    img[top:bottom, left:right, :].fill(mask_val)

    img = Image.fromarray(img)

    return img


class Cutout(object):

    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        img = np.array(img)

        mask_val = img.mean()

        top = np.random.randint(0 - self.length//2, img.shape[0] - self.length)
        left = np.random.randint(0 - self.length//2, img.shape[1] - self.length)
        bottom = top + self.length
        right = left + self.length

        top = 0 if top < 0 else top
        left = 0 if left < 0 else top

        img[top:bottom, left:right, :] = mask_val

        img = Image.fromarray(img)

        return img
