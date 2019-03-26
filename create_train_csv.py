from glob import glob
import fnmatch
import random
import matplotlib.pylab as plt
import cv2
import numpy as np
import pandas as pd
import csv


def main():
    imagePatches = glob('IDC_regular_ps50_idx5/**/*.png', recursive=True)

    print('Total All Images: {}'.format(len(imagePatches)))

    patternZero = '*class0.png'
    patternOne = '*class1.png'
    classZero = fnmatch.filter(imagePatches, patternZero)
    classOne = fnmatch.filter(imagePatches, patternOne)

    X,Y = proc_images(imagePatches, classZero, classOne, 0, len(imagePatches))

    df = pd.DataFrame()
    df["images"] = X
    df["labels"] = Y
    X2 = df["images"]
    Y2 = df["labels"]
    X2 = np.array(X2)

    describeData(X2, Y2)

    dict_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
    print(df.head(10))
    print("")
    print(dict_characters)

def describeData(a,b):
    print('Total number of images: {}'.format(len(a)))
    print('Number of IDC(-) Images: {}'.format(np.sum(b==0)))
    print('Number of IDC(+) Images: {}'.format(np.sum(b==1)))
    print('Percentage of positive images: {:.2f}%'.format(100*np.mean(b)))

def proc_images(imagePatches, classZero, classOne, lowerIndex,upperIndex):
    """
    Returns two arrays:
        x is an array of image path
        y is an array of labels
    """
    x = []
    y = []
    with open('train.csv', 'w', newline='') as csvfile:
        fileWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for img in imagePatches[lowerIndex:upperIndex]:
            x.append(img)
            if img in classZero:
                y.append(0)
                fileWriter.writerow([img, 0])
            elif img in classOne:
                y.append(1)
                fileWriter.writerow([img, 1])
            else:
                return
    return x,y

def plotImage(image_name):
    image = cv2.imread(image_name)
    image = cv2.resize(image, (50,50))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    return

def plotMultipleImages(bunchOfImages):
    i_ = 0
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    for l in bunchOfImages[:25]:
        im = cv2.imread(l)
        im = cv2.resize(im, (50, 50))
        plt.subplot(5, 5, i_ + 1)  # .set_title(l)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB));
        plt.axis('off')
        i_ += 1
    plt.show()
    return

def randomImages(a):
    r = random.sample(a, 4)
    plt.figure(figsize=(16,16))
    plt.subplot(131)
    plt.imshow(cv2.imread(r[0]))
    plt.subplot(132)
    plt.imshow(cv2.imread(r[1]))
    plt.subplot(133)
    plt.imshow(cv2.imread(r[2]))
    plt.show()
    return

if __name__ == '__main__':
    main()