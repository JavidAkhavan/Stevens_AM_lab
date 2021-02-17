# python test.py --dataset "set_name" --neighbors "# of neighors"


# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import pickle
import joblib
import time


def image_to_feature_vector(image, name, size=(128, 128)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(32, 32, 32)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] handling images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label
    # our images were named as labels.image_number.format
    image = cv2.imread(imagePath)
    if image is None:
        # print('skip')
        continue

    # get the labels from the name of the images by extract the string before "."
    label = imagePath.split(os.path.sep)[-2]
    # label = imagePath.split(os.path.sep)[-1].split(".")[0].split('_')[:-1]
    # label = "_".join(label)

    # extract raw pixel intensity "features"
    # followed by a color histogram to characterize the color distribution of the pixels
    # in the image
    pixels = image_to_feature_vector(image, imagePath, size=(30, 30))
    hist = extract_color_histogram(image)

    # add the messages we got to the raw images, features, and labels matricies
    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)

    # show an update every 200 images until the last image
    if i > 0 and ((i + 1) % 200 == 0 or i == len(imagePaths) - 1):
        print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
    rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
    features.nbytes / (1024 * 1000.0)))

# partition the data into training and testing splits, using 85%
# of the data for training and the remaining 15% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
    rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    features, labels, test_size=0.25, random_state=42)

# # k-NN
# print("\n")
# print("[INFO] evaluating raw pixel accuracy...")
# model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"], metric='cityblock')
# model.fit(trainRI, trainRL)
# acc = model.score(testRI, testRL)
# print("[INFO] k-NN classifier: k=%d" % args["neighbors"])
# print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
# # save the model to disk
# filename = 'KNN_raw_19_' + time.strftime("%Y%m%d_%H%M%S") + '.sav'
# joblib.dump(model, open(filename, 'wb'))
# print("[INFO] model saved as " + filename)
#
# # k-NN
# print("\n")
# print("[INFO] evaluating histogram accuracy...")
# model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"], metric='cityblock')
# model.fit(trainFeat, trainLabels)
# acc = model.score(testFeat, testLabels)
# print("[INFO] k-NN classifier: k=%d" % args["neighbors"])
# print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
# # save the model to disk
# filename = 'KNN_hist_19_' + time.strftime("%Y%m%d_%H%M%S") + '.sav'
# joblib.dump(model, open(filename, 'wb'))
# print("[INFO] model saved as " + filename)

#
#neural network
# print("\n")
# print("[INFO] evaluating raw pixel accuracy...")
# start = time.time()
# model = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000, alpha=1e-4,
#                       solver='adam', tol=1e-4, random_state=1,
#                       learning_rate_init=.1) # max_iter=1000,
# model.fit(trainRI, trainRL)
# print("[INFO] training took {:.2f} seconds".format(time.time() - start))
# # acc = model.score(testRI, testRL)
# testRL_pred = model.predict(testRI)
# acc = f1_score(testRL, testRL_pred, average='micro')
# print("[INFO] neural network raw pixel accuracy: {:.2f}".format(acc))
# # save the model to disk
# filename = 'NN_raw_' + time.strftime("%Y%m%d_%H%M%S") + '.sav'
# joblib.dump(model, open(filename, 'wb'))
# print("[INFO] model saved as " + filename)
#
# #neural network
# print("\n")
# print("[INFO] evaluating histogram accuracy...")
# model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
#                       solver='sgd', tol=1e-4, random_state=1,
#                       learning_rate_init=.1)
# model.fit(trainFeat, trainLabels)
# acc = model.score(testFeat, testLabels)
# print("[INFO] neural network histogram accuracy: {:.2f}%".format(acc * 100))
# filename = 'NN_dist_' + time.strftime("%Y%m%d_%H%M%S") + '.sav'
# joblib.dump(model, open(filename, 'wb'))
# print("[INFO] model saved as " + filename)

# #SVC
print("\n")
print("[INFO] evaluating raw pixel accuracy...")
start = time.time()
model = SVC(C=5, class_weight='balanced')  #max_iter=2000,
model = joblib.load('svc_raw_5_20210215_201023.sav')
model.fit(trainRI, trainRL)
print("[INFO] training took {:.2f} seconds".format(time.time() - start))
acc = model.score(testRI, testRL)
testRL_pred = model.predict(testRI)
# acc = f1_score(testRL, testRL_pred, average='micro')
print("[INFO] SVM-SVC raw pixel accuracy: {:.2f}%".format(acc * 100))
# target_names = ['Empty','OK', 'Over', 'Under']
print(classification_report(testRL, testRL_pred))
# filename = 'svc_raw_5_' + time.strftime("%Y%m%d_%H%M%S") + '.sav'
# joblib.dump(model, open(filename, 'wb'))
# print("[INFO] model saved as " + filename)

#SVC
# print("\n")
# print("[INFO] evaluating histogram accuracy...")
# model = SVC(C=0.5, max_iter=1000, class_weight='balanced')
# model.fit(trainFeat, trainLabels)
# acc = model.score(testFeat, testLabels)
# print("[INFO] SVM-SVC histogram accuracy: {:.2f}%".format(acc * 100))
# filename = 'svc_dist_' + time.strftime("%Y%m%d_%H%M%S") + '.sav'
# joblib.dump(model, open(filename, 'wb'))
# print("[INFO] model saved as " + filename)
# '''
# '''

# # load the model from disk
# loaded_model = joblib.load(filename)
# result = loaded_model.score(X_test, Y_test)
# print(result)