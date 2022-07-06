import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from Pipeline import Process

class Filter(Process):
    def __init__(self, f, bound_method='crop', parallel_method='pixel', workers=1):
        self.f = np.array(f)
        self.bound_method = bound_method
        self.workers = workers
        
        if parallel_method in ['pixel', 'row']:
            self.parallel_method = parallel_method
        else:
            raise ValueError('Wrong parallel method')
    
    def run(self, data):
        self.data = data
        self.res_shape = [data.shape[0] - self.f.shape[0] + 1, data.shape[1] - self.f.shape[1] + 1] + list(data.shape[2:])
        
        if self.parallel_method == 'pixel':
            total_pixels = self.res_shape[0] * self.res_shape[1]
            pixels = [(i,j) for i in range(self.res_shape[0]) for j in range(self.res_shape[1])]
            routine = partial(self.apply_to_pixel, data = data)
            with Pool(self.workers) as p:
                res = np.array(list(tqdm(p.imap(routine, pixels), total=total_pixels))).reshape(self.res_shape)
                
        elif self.parallel_method == 'row':
            total_rows = self.res_shape[0]
            rows = [i for i in range(self.res_shape[0])]
            routine = partial(self.apply_to_row, data = data)
            with Pool(self.workers) as p:
                res = np.array(list(tqdm(p.imap(routine, rows), total=total_rows)))
                
        else:
            raise ValueError('Wrong parallel method')
            
        return res
    
    ### TOO SLOW ###
    def apply_to_pixel(self, pixel, data):
        i, j = pixel
        res = []
        for color in range(self.res_shape[2]):
            res.append(sum((data[i:i+self.f.shape[0],j:j+self.f.shape[1], color] * self.f).flatten()))
        return res
    
    def apply_to_row(self, row, data):
        res = np.zeros((self.res_shape[1], self.res_shape[2]))
        for j in range(self.res_shape[1]):
            for color in range(self.res_shape[2]):
                res[j][color] = sum((data[row:row+self.f.shape[0],j:j+self.f.shape[1], color] * self.f).flatten())
        return res
    
class Normalize_Image(Process):
    def __init__(self, negative_correction_method='abs'):
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
        
        #reencode as uint8
        return data.astype(np.uint8)
    
class Reverse_Image(Process):
    def run(self, data):
        return 255 - data