class Process:
    def __init__(self):
        pass
    
    def run(self, data):
        return data

class Pipeline:
    def __init__(self, processes):
        self.processes = processes
        
    def process(self, image):
        output = image
        for process in self.processes:
            output = process.run(output)
        return output