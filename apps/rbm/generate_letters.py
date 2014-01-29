import Image
import ImageFont
import ImageDraw
import numpy as np


def generate_letters():

    font = ImageFont.truetype("courbd.ttf", 35)

    imgs = []
    for lnum in range(26):
        l = chr(ord('A') + lnum)

        img = Image.new('L', (28,28), 0)
        draw = ImageDraw.Draw(img)

        draw.text((3,-5), l, font=font, fill=255)
        a = np.asarray(img)
        imgs.append(np.asarray(img))

    return imgs
    

def generate_letter_dataset():
    imgs = generate_letters()

    ds = np.zeros((len(imgs), imgs[0].shape[0] * imgs[0].shape[1]))
    for i in range(len(imgs)):
        ds[i, :] = np.reshape(imgs[i], (-1,)) / 255.

    return ds

