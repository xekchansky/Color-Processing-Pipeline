import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

def Filter_Visualization(f):
    x, y = np.mgrid[-f.shape[0] // 2 + 1 : f.shape[0] // 2 + 1, -f.shape[1] // 2 + 1 : f.shape[1] // 2 + 1]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, f, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()
    
def Image_Result_Comparison(image, res):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax2.imshow(res)
    plt.show()