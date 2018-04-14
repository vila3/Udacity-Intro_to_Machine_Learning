import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imsave
#from sklearn.datasets import fetch_lfw_people
import random, string
from time import time
from itertools import product


def weighted_average(pixel):
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]


def gray_loop(image_arr):
    gray = np.zeros((image_arr.shape[0], image_arr.shape[1]))  # init 2D numpy array

    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
            gray[rownum][colnum] = weighted_average(image[rownum][colnum])*255

    return gray


def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))


t0 = time()
image = mpimg.imread('10649067_607172606054894_2989485725976337220_o.png')
print time() - t0

# lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# n_samples, h, w = lfw_people.images.shape
# print lfw_people.data[0].reshape((h, w))
# print lfw_people.images
# print '-----'
# print len(lfw_people.data)
# print len(lfw_people.data[0])
# print img
# print gray

t0 = time()
gray = gray_loop(image)
print time() - t0

t0 = time()
imsave(randomword(10) + '.png', gray)
print time() - t0
# plt.imshow(gray, cmap=plt.cm.gray)
# plt.show()
