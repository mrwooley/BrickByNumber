# Author: Megan Wooley
# GitHub username: mrwooley
# Date: 2/25/2024
# Description: *

from PIL import Image, ImageOps, ImageColor, ImageEnhance, ImageFilter
import numpy as np
import math
import matplotlib.pyplot as plt
import re


class BrickByNumber:
    def __init__(self, img_file, color_space='bricklink-rgb'):
        self._color_space = color_space
        self._img_file = img_file

        with open(f'mapped-color-spaces/{color_space}.npy', 'rb') as f:
            self._colormap = np.load(f)

        with Image.open(img_file) as img:
            img.load()

        self.img = img

    def recolor_image(self, img, preview=True):
        # Recolor image
        img_rgb = np.asarray(img)
        img_size = img_rgb.shape
        old_color = img_rgb.reshape(img_size[0] * img_size[1], 3)
        r, g, b = old_color[:, 0], old_color[:, 1], old_color[:, 2]
        new_color = self._colormap[:, r, g, b].T
        recolored_img = Image.fromarray(
            new_color.reshape(img_size).astype(np.uint8))

        if preview:
            recolored_img.show()

        return recolored_img

    def resize_image(self, img, width=12, preview=True):
        # Resize image
        img = img.resize((width * 16,
                                  round(img.height / img.width * width * 16)),
                                 Image.NEAREST)

        if preview:
            self.preview(img, width)

        return img

    def preview(self, img, width=3, bg_color='black', save=True):
        color = np.asarray(img)

        # Plotly Version
        x = range(img.width)
        y = range(img.height)
        xv, yv = np.meshgrid(x, y)

        plt.rcParams['figure.facecolor'] = bg_color
        plt.rcParams['axes.facecolor'] = bg_color
        plt.rcParams['text.color'] = "white"

        plt.figure(
            figsize=(2 * width, math.ceil(2 * width * img.height / img.width)))

        plt.scatter(
            x=xv.flatten(),
            y=yv.flatten(),
            c=[(c[0] / 255, c[1] / 255, c[2] / 255) for c in color.reshape(
                (img.height * img.width, 3))],
            clip_on=False,
        )
        plt.ylim(max(y) + 2, min(y) - 2)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        plt.title(f'Img: {self._img_file} Recolor: {self._color_space}')

        if save:
            img_name = self._img_file
            ind = img_name.rfind('.')
            img_name = img_name[:ind]
            ind = img_name.rfind('/')
            img_name = img_name[ind+1:]

            plt.savefig(f'local/{img_name}-{self._color_space}.png')

        plt.show()


if __name__ == "__main__":
    bbn = BrickByNumber('images/nighthawks3.jpg', 'bricklink-rgb')
    resized_img = bbn.resize_image(bbn.img)
    recolored_img = bbn.recolor_image(resized_img)
