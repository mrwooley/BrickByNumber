# Author: Megan Wooley
# GitHub username: mrwooley
# Date: 2/28/2024
# Description: *


import json

import numpy as np


def save_array(arr, save_name):
    with open(save_name, 'wb') as f:
        np.save(f, arr)


class ColorSpaceMapping:
    """Color space transformations"""

    def __init__(self, rgb_map):
        self._rgb_map = rgb_map

    def convert_rgb(self, color_space):
        """
        Convert RGB space to RGB, CIEXYZ, or CIELAB color spaces.

        :param color_space: options are 'rgb', 'xyz', 'lab'
        :return: transformed color map
        """
        rgb_map = self._rgb_map.copy()
        if color_space == 'rgb':
            return rgb_map
        elif color_space == 'xyz':
            return self._rgb2xyz(rgb_map, 1).transpose()
        elif color_space == 'lab':
            xyz_map = self._rgb2xyz(rgb_map, 1)
            return self._xyz2lab(xyz_map).transpose()
        else:
            raise Exception('Invalid color space option')

    def get_color_space_matrix(self, color_space):
        """
        Get full color space matrix. Converted from RGB 256x256x256 space.

        :param color_space: options are 'rgb', 'xyz', or 'lab'
        :return: 256x256x256 matrix
        """
        rgb_matrix = np.meshgrid(range(256), range(256), range(256),
                                    indexing='ij')
        if color_space == 'rgb':
            return rgb_matrix
        elif color_space == 'xyz':
            return self._rgb2xyz(rgb_matrix)
        elif color_space == 'lab':
            xyz_matrix = self._rgb2xyz(rgb_matrix)
            return self._xyz2lab(xyz_matrix)
        else:
            raise Exception('Invalid color space option')

    def get_color_difference_formula(self, color_space, calc):
        """
        Choose color space formula by checking color space and calculation
        type.

        :param color_space: options are 'rgb', 'xyz', 'lab'
        :param calc: options are 'euclidean', 'adjusted-euclidean',
        and 'ciede200'
        :return: function for the color difference formula
        """
        if color_space == 'rgb' and calc == 'adjusted-euclidean':
            return self._adjusted_euclidean_distance

        elif color_space == 'lab' and calc == 'ciede2000':
            return self._ciede2000

        elif calc == 'euclidean':
            return self._euclidean_distance

        else:
            raise Exception('Invalid calculation option')

    def remap_color_space(self, color_space='rgb', calc='euclidean'):
        """
        Remap a color space to the nearest color in a provided list of colors.

        :param color_space: options are 'rgb', 'xyz', 'lab'
        :param calc: options are 'euclidean', 'adjusted-euclidean',
        and 'ciede200'
        :return:
        """
        color_map = self.convert_rgb(color_space)
        color_space_matrix = self.get_color_space_matrix(color_space)
        color_diff_formula = self.get_color_difference_formula(color_space,
                                                               calc)

        mapped_color_space = [np.zeros((256, 256, 256)),
                           np.zeros((256, 256, 256)),
                           np.zeros((256, 256, 256))]
        prev_dist = np.ones((256, 256, 256)) * np.inf
        for ind in range(len(rgb_map)):
            print(f'{ind + 1} of {len(rgb_map)}')
            # Calculate color difference for the entire space from a new color
            color = color_map[ind]
            dist = color_diff_formula(color_space_matrix, color)

            # Make updates for any distances that are an improvement
            mask = dist < prev_dist
            mapped_color_space[0][mask] = rgb_map[ind][0]
            mapped_color_space[1][mask] = rgb_map[ind][1]
            mapped_color_space[2][mask] = rgb_map[ind][2]
            prev_dist[mask] = dist[mask]

        return mapped_color_space

    @staticmethod
    def _rgb2xyz(rgb, axis=0, adobe=False):
        """
        Convert RGB to XYZ color space.
        https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz

        :param rgb:
        :param axis:
        :param adobe:
        :return:
        """
        srgb = np.array(rgb) / 255
        if adobe:
            srgb_p = np.power(srgb, 2.19921875)

            # Adobe RGB to XYZ conversion matrix
            matrix = np.array([[0.5767309, 0.1855540, 0.1881852],
                               [0.2973769, 0.6273491, 0.0752741],
                               [0.0270343, 0.0706872, 0.9911085]])
        else:
            srgb_p = np.where(srgb > 0.04045, ((srgb + 0.055) / 1.055) ** 2.4,
                              srgb / 12.92) * 100

            matrix = np.array([[0.412453, 0.357580, 0.180423],
                               [0.212671, 0.715160, 0.072169],
                               [0.019334, 0.119193, 0.950227]])

        xyz = np.tensordot(matrix, srgb_p, axes=(1, axis))

        return xyz

    @staticmethod
    def _xyz2lab(xyz):
        """
        Convert XYZ to CIELAB color space.
        https://en.wikipedia.org/wiki/CIELAB_color_space

        :param xyz:
        :return:
        """
        # Standard Illuminant D65
        xn, yn, zn = 95.0489, 100, 108.8840

        s_xyz = xyz / np.array([xn, yn, zn])
        delta = 6 / 29

        sxyz_p = np.where(s_xyz > delta ** 3,
                          np.cbrt(s_xyz),
                          (s_xyz * (delta ** -2)) / 3 + 4 / 29)

        L = 116 * sxyz_p[1] - 16
        a = 500 * (sxyz_p[0] - sxyz_p[1])
        b = 200 * (sxyz_p[1] - sxyz_p[2])
        lab = np.array([L, a, b])
        return lab

    @staticmethod
    def _ciede2000(lab1, lab2):
        """
        Calculate color difference using the CIEDE2000 formula.
        https://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
        :param lab1:
        :param lab2:
        :return:
        """
        # Calculate C and h prime
        C1 = np.sqrt(lab1[1] ** 2 + lab1[2] ** 2)
        C2 = np.sqrt(lab2[1] ** 2 + lab2[2] ** 2)
        C_bar = (C1 + C2) / 2

        G = 0.5 * (1 - np.sqrt(C_bar ** 7 / (C_bar ** 7 + 25 ** 7)))
        a1_p = (1 + G) * lab1[1]
        a2_p = (1 + G) * lab2[1]
        C1_p = np.sqrt(a1_p ** 2 + lab1[2] ** 2)
        C2_p = np.sqrt(a2_p ** 2 + lab2[2] ** 2)

        h1_p = np.where(np.logical_and(lab1[2] == 0, a1_p == 0), 0, np.degrees(
            np.arctan2(lab1[2], a1_p)))
        h2_p = np.where(np.logical_and(lab2[2] == 0, a2_p == 0), 0, np.degrees(
            np.arctan2(lab2[2], a2_p)))

        # Calculate delta L, C, and H
        L_delta_p = lab2[0] - lab1[0]
        C_delta_p = C2_p - C1_p
        h_delta = np.where(abs(h2_p - h1_p) <= 180, h2_p - h1_p,
                           (h2_p - h1_p) + 360)
        h_delta = np.where((h2_p - h1_p) > 180, (h2_p - h1_p) - 360, h_delta)
        h_delta = np.where(C1_p * C2_p == 0, 0, h_delta)

        H_delta_p = 2 * np.sqrt(C1_p * C2_p) * np.degrees(np.sin(h_delta / 2))

        # Calculate color difference
        L_bar_p = (lab1[0] + lab2[0]) / 2
        C_bar_p = (C1_p + C2_p) / 2

        h_bar_p = np.where(abs(h2_p - h1_p) <= 180, (h1_p + h2_p) / 2,
                           (h1_p + h2_p + 360) / 2)
        h_bar_p = np.where(
            np.logical_and(abs(h2_p - h1_p) > 180, (h1_p + h2_p) > 360),
            (h1_p + h2_p - 360) / 2, h_bar_p)
        h_bar_p = np.where(C1_p * C2_p == 0, h1_p + h2_p, h_bar_p)

        T = 1 - 0.17 * np.degrees(np.cos(h_bar_p - 30)) + 0.24 * np.degrees(
            np.cos(
                2 * h_bar_p)) + 0.32 * np.degrees(
            np.cos(3 * h_bar_p + 6)) - 0.2 * np.degrees(
            np.cos(4 * h_bar_p - 63))

        theta_delta = 30 * np.exp(-((h_bar_p - 275) / 25) ** 2)
        R_C = 2 * np.sqrt(C_bar_p ** 7 / (C_bar_p ** 7 + 25 ** 7))
        S_L = 1 + 0.015 * (L_bar_p - 50) ** 2 / np.sqrt(20 + (L_bar_p - 50))
        S_C = 1 + 0.045 * C_bar_p
        S_H = 1 + 0.015 * C_bar_p * T
        R_T = -np.degrees(np.sin(2 * theta_delta)) * R_C

        k_L = 1
        k_C = 1
        k_H = 1
        color_diff = np.sqrt(
            (L_delta_p / (k_L * S_L)) ** 2 + (C_delta_p / (k_C * S_C)) ** 2 +
            (H_delta_p / (k_H * S_H)) ** 2 +
            R_T * (C_delta_p / (k_C * S_C)) * (H_delta_p / (k_H * S_H)))

        return color_diff

    @staticmethod
    def _adjusted_euclidean_distance(rgb1, rgb2):
        """
        Adjusted for 'redmean'
        :param rgb1:
        :param rgb2:
        :return:
        """
        rbar = (rgb2[0] + rgb1[0]) / 2
        dist = np.sqrt(
            (2 + rbar / 256) * (rgb2[0] - rgb1[0]) ** 2 +
            4 * (rgb2[1] - rgb1[1]) ** 2 +
            (2 + (255 - rbar) / 256) * (
                    rgb2[2] - rgb1[2]) ** 2)

        return dist

    @staticmethod
    def _euclidean_distance(color1, color2):
        """

        :param color1:
        :param color2:
        :param adjusted:
        :return:
        """
        dist = np.sqrt((color2[0] - color1[0]) ** 2 +
                       (color2[1] - color1[1]) ** 2 +
                       (color2[2] - color1[2]) ** 2)

        return dist



if __name__ == '__main__':
    # pieces_filename = 'BrickLinkColors.xlsx'
    # df = pd.read_excel(pieces_filename, usecols="C:D",
    #                    names=['hex', 'name'])
    #
    # df['rgb'] = df.hex.map(lambda color: ImageColor.getrgb(color))
    # df.to_json('bricklink_colormap.json')

    colormap_json = 'bricklink_colormap.json'
    with open(colormap_json, 'r') as f:
        colormap = json.load(f)

    rgb_map = np.array(list(colormap['rgb'].values()))

    cs = ColorSpaceMapping(rgb_map)
    rgb_adjusted_euclidean_mapping = cs.remap_color_space('rgb', calc='adjusted-euclidean')
    rgb_euclidean_mapping = cs.remap_color_space('rgb')
    xyz_euclidean_mapping = cs.remap_color_space('xyz')
    lab_euclidean_mapping = cs.remap_color_space('lab')
    lab_ciede2000_mapping = cs.remap_color_space('lab', calc='ciede2000')
