import numpy as np
import glob
import matplotlib.pyplot as plt

if __name__=="__main__":
    for file in glob.glob('*.txt'):
        array = np.loadtxt(file)
        fig = plt.imshow(array)
        plt.colorbar()
        plt.savefig(file.split('.')[0]+ '.png')
        plt.close()
