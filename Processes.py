import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from Pipeline import Process


class Filter(Process):
    def __init__(self, f, bound_method='crop', parallel_method='pixel', workers=1):
        super().__init__()
        self.f = np.array(f)
        self.bound_method = bound_method
        self.workers = workers

        if parallel_method in ['pixel', 'row']:
            self.parallel_method = parallel_method
        else:
            raise ValueError('Wrong parallel method')

    def run(self, data):
        res_shape = [data.shape[0] - self.f.shape[0] + 1, data.shape[1] - self.f.shape[1] + 1] + list(
            data.shape[2:])

        if self.parallel_method == 'pixel':
            total_pixels = res_shape[0] * res_shape[1]
            pixels = [(i, j) for i in range(res_shape[0]) for j in range(res_shape[1])]
            routine = partial(self.apply_to_pixel, data=data, res_shape=res_shape)
            with Pool(self.workers) as p:
                res = np.array(list(tqdm(p.imap(routine, pixels), total=total_pixels))).reshape(res_shape)

        elif self.parallel_method == 'row':
            total_rows = res_shape[0]
            rows = [i for i in range(res_shape[0])]
            routine = partial(self.apply_to_row, data=data, res_shape=res_shape)
            with Pool(self.workers) as p:
                res = np.array(list(tqdm(p.imap(routine, rows), total=total_rows)))

        else:
            raise ValueError('Wrong parallel method')

        return res

    # TOO SLOW
    def apply_to_pixel(self, pixel, data, res_shape):
        i, j = pixel
        res = []
        for color in range(res_shape[2]):
            res.append(sum((data[i:i + self.f.shape[0], j:j + self.f.shape[1], color] * self.f).flatten()))
        return res

    def apply_to_row(self, row, data, res_shape):
        res = np.zeros((res_shape[1], res_shape[2]))
        for j in range(res_shape[1]):
            for color in range(res_shape[2]):
                res[j][color] = sum((data[row:row + self.f.shape[0], j:j + self.f.shape[1], color] * self.f).flatten())
        return res


class BilateralFilter(Process):
    def __init__(self, diameter, sigma_i, sigma_s):
        super().__init__()
        self.diameter = diameter
        self.sigma_i = sigma_i
        self.sigma_s = sigma_s

    @staticmethod
    def gaussian(x, sigma):
        return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))

    @staticmethod
    def distance(x1, y1, x2, y2):
        return np.sqrt(np.abs((x1 - x2) ** 2 - (y1 - y2) ** 2))

    def run(self, data):
        res = np.zeros(data.shape)

        for row in tqdm(range(len(data))):
            for col in range(len(data[0])):
                wp_total = 0
                filtered_data = 0
                for k in range(self.diameter):
                    for l in range(self.diameter):
                        n_x = row - (self.diameter / 2 - k)
                        n_y = col - (self.diameter / 2 - l)
                        if n_x >= len(data):
                            n_x -= len(data)
                        if n_y >= len(data[0]):
                            n_y -= len(data[0])
                        gi = self.gaussian(data[int(n_x)][int(n_y)] - data[row][col], self.sigma_i)
                        gs = self.gaussian(self.distance(n_x, n_y, row, col), self.sigma_s)
                        wp = gi * gs
                        filtered_data = filtered_data + (data[int(n_x)][int(n_y)] * wp)
                        wp_total += wp
                filtered_data = filtered_data // wp_total
                res[row][col] = (np.round(filtered_data)).astype(int)
        return res


class NormalizeImage(Process):
    def __init__(self, negative_correction_method='abs'):
        super().__init__()
        if negative_correction_method in ['abs', 'shift_to_zero']:
            self.neg_method = negative_correction_method
        else:
            raise ValueError('Wrong negative correction method')

    def run(self, data):
        data = np.array(data)

        # negative correction
        if self.neg_method == 'abs':
            data = np.abs(data)
        elif self.neg_method == 'shift_to_zero':
            data -= data.min()
        else:
            raise ValueError('Wrong negative correction method')

        if data.max() == 0:
            raise ValueError('Image max value is 0')

        # normalization to [0,255]
        data /= data.max() / 255.0

        # reencode as uint8
        return data.astype(np.uint8)


class ReverseImage(Process):
    def run(self, data):
        return 255 - data
