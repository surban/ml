import images2gif
from tempfile import mktemp
from PIL import Image
import os


class GIFCreator(object):
    def __init__(self):
        self.img_paths = []

    def add_image(self):
        filename = mktemp(suffix='.png')
        # print filename
        self.img_paths.append(filename)
        return filename

    def create(self, filename, duration=0.1, repeat=True, dither=False):
        imgs = []
        files = []
        for fn in self.img_paths:
            f = open(fn, 'rb')
            files.append(f)
            img = Image.open(f)
            imgs.append(img)

        images2gif.writeGif(filename, imgs, duration=duration, repeat=repeat, dither=dither)

        del imgs
        for f in files:
            f.close()
        for fn in self.img_paths:
            os.remove(fn)
        self.img_paths = []


