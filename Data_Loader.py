import os
from PIL import Image
import numpy as np

class DataIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._index = 0
    def __next__(self):
        if self._index < len(self.data_loader.files):
            result =  self.data_loader.get(self._index)
            self._index += 1
            return result
        raise StopIteration

class Data_Loader:
    def __init__(self, input_path, output_path=None):
        self.input_path = input_path
        self.output_path = output_path
        
        self.file_names = os.listdir(self.input_path)
        
        #ignore ipynb_checkpoints:
        self.file_names.remove('.ipynb_checkpoints')
        
        self.files = [os.path.join(self.input_path, file) for file in self.file_names]
        
    def get_len(self):
        return len(self.files)
    
    def get(self, i):
        return np.asarray(Image.open(self.files[i]))
    
    def write(self, i, data):
        name = self.file_names[i]
        Image.fromarray(data).save(os.path.join(self.output_path, name))
        
    def __iter__(self):
        return DataIterator(self)