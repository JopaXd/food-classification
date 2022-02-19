from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

X,Y = [], []

trainX = []
trainY = []
valX = []
valY = []
testX = []
testY = []

with open("/media/pavle/HDD_disk/deep-learning/food-classification/food-101/meta/train.txt") as f:
	trainData = f.readlines()

for trainItem in trainData:
	trainItem = trainItem.strip()
	label = trainItem.split("/")[0]
	X.append(f"/media/pavle/HDD_disk/deep-learning/food-classification/food-101/images/{trainItem}.jpg")
	Y.append(label)

(trainX, valX, trainY, valY) = train_test_split(X, Y, test_size=0.25, random_state=42)

with open("/media/pavle/HDD_disk/deep-learning/food-classification/food-101/meta/test.txt") as f:
	testData = f.readlines()

for testItem in testData:
	testItem = testItem.strip()
	label = testItem.split("/")[0]
	testX.append(f"/media/pavle/HDD_disk/deep-learning/food-classification/food-101/images/{testItem}.jpg")
	testY.append(label)

le = LabelEncoder()
trainY = le.fit_transform(trainY)
valY = le.fit_transform(valY)
testY = le.fit_transform(testY)

DATASET_PATH = "/media/pavle/HDD_disk/deep-learning/food-classification/food-101"

trainWriter = HDF5DatasetWriter((len(trainX), 128, 128, 3), f"{DATASET_PATH}/train.hdf5")
valWriter = HDF5DatasetWriter((len(valX), 128, 128, 3), f"{DATASET_PATH}/validate.hdf5")
testWriter = HDF5DatasetWriter((len(testX), 128, 128, 3), f"{DATASET_PATH}/test.hdf5")

print("Creating train dataset...")

for (path, label) in zip(trainX, trainY):
	image = cv2.imread(path)
	image = cv2.resize(image, (128,128))
	trainWriter.add([image], [label])

print("Creating validation dataset...")

for (path, label) in zip(valX, valY):
	image = cv2.imread(path)
	image = cv2.resize(image, (128,128))
	valWriter.add([image], [label])

print("Creating test dataset...")

for (path, label) in zip(testX, testY):
	image = cv2.imread(path)
	image = cv2.resize(image, (128,128))
	testWriter.add([image], [label])

print("Done!")