import numpy as np
from tqdm import tqdm

from Pipeline import Process

class Filter(Process):
    def __init__(self, f, bound_method='crop'):
        self.f = np.array(f)
    
    def run(self, data):
        shape = [data.shape[0] - self.f.shape[0] + 1, data.shape[1] - self.f.shape[1] + 1] + list(data.shape[2:])
        res = np.zeros(shape)
        for i in tqdm(range(data.shape[0] - self.f.shape[0] + 1)):
            for j in range(data.shape[1] - self.f.shape[1] + 1):
                for color in range(shape[2]):
                    res[i][j][color] = sum((data[i:i+self.f.shape[0],j:j+self.f.shape[1], color] * self.f).flatten())
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
            
        # normalization to [0,255]
        data /= data.max() / 255.0
        
        #reencode as uint8
        return data.astype(np.uint8)
    
class Reverse_Image(Process):
    def run(self, data):
        return 255 - data