import numpy as np
from matplotlib import pyplot as plt

def show_ind():
    i = 0
    while i < 150:
        img = np.load("Datasets/SrS/0_" + str(i) + ".npy")
        plt.imshow(img, cmap='inferno')
        plt.show()
        i += 5
    

def show_all():
    fig, axes = plt.subplots(15,10, figsize=(15,10))
    for i, ax in enumerate(axes.ravel()): 
        img = np.load("Datasets/SrS/0_" + str(i) + ".npy")
        ax.imshow(img, cmap='gray')
        ax.axis('off')
#show_all()
show_ind()
plt.tight_layout()


plt.show()