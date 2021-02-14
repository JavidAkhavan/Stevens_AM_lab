'''
Created on 04/06/2020

Build a CNN model to classification the image from point cloud

'''

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
# from sklearn.model_selection import train_test_split
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# %% data generator
'''
 Use keras to import all the training data. Each folder represent one category.
'''
datagen = keras.preprocessing.image.ImageDataGenerator(
    # rotation_range=90,  # Degree range for random rotations.
    # width_shift_range=0, # fraction of total width
    # height_shift_range=0, # fraction of total height
    # shear_range=0, # ramdom change
    zoom_range=[1.0, 1.5],  # ramdom zoom
    horizontal_flip=True,  # flip transform
    # fill_mode='nearest',
    # brightness_range=[0.8, 1.2],
    # zca_whitening = True,
    validation_split=0.3
)  #
# 'D:/Google Drive/Research/Laser Scanner/Control/Data/3-24-20/total/CNN_6/rgb/'
train_generator = datagen.flow_from_directory(
    'Test_data_labeled/labeled/',
    # target_size=(411, 411),
    # target_size=(124, 124),
    target_size=(32, 32),
    color_mode='rgb',
    batch_size=150,
    class_mode='binary',
    subset='training',

)
# 'D:/Google Drive/Research/Laser Scanner/Control/Data/3-24-20/total/CNN_6/rgb/'
validation_generator = datagen.flow_from_directory(
    'Test_data_labeled/labeled/',
    #    target_size=(411, 411),
    target_size=(32, 32),
    color_mode='rgb',
    batch_size=100,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications import VGG19
# from tensorflow.keras.applications import MobileNetV2
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(32, 32, 3))
# trainable_layer_names = ['block5_conv1', 'block5_conv2','block5_conv3', 'block5_pool']
# conv_base.trainable = True
# for layer in conv_base.layers:
#     if layer.name in trainable_layer_names:
#         layer.trainable = True
#     else:
#         layer.trainable = False

conv_base.summary()
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))
conv_base.trainable = False
model.summary()
#################################################################

# %% My CNN
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.BatchNormalization(scale=False, center=False))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.25))
# model.add(layers.BatchNormalization(scale=False, center=False))
#
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.Dropout(0.25))
# model.add(layers.BatchNormalization(scale=False, center=False))
#
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.25))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.BatchNormalization(scale=False, center=False))
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Dense(4, activation='softmax'))
# model.summary()

model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']  # ,f1_m,precision_m, recall_m
              ) #'adam'

history = model.fit(
    train_generator,
    # steps_per_epoch=30,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=45
)

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, 150)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm1 = confusion_matrix(validation_generator.classes, y_pred)
print(cm1)
print('Classification Report')
target_names = ['OK', 'Over', 'Under', 'Under ++']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

TruePositive = np.diag(cm1)
print('TruePositive')
print(TruePositive)

FalsePositive = []
for i in range(3):
    FalsePositive.append(sum(cm1[:, i]) - cm1[i, i])
print('FalsePositive')
print(FalsePositive)

FalseNegative = []
for i in range(3):
    FalseNegative.append(sum(cm1[i, :]) - cm1[i, i])
print('FalseNegative')
print(FalseNegative)

# # Save the weights
# model.save_weights('cnn_model/model_weights_6_19.h5')
#
# # Save the model architecture
# with open('cnn_model/model_architecture.json', 'w') as f:
#     f.write(model.to_json())

# img = Image.open('D:/Google Drive/Research/Laser Scanner/Control/Data/3-24-20/2/002/6.png')
# img = img.resize([124, 124])
# data_array = np.array(img).reshape(1, 124, 124, 3)
# import cv2
# img_cv = cv2.imread('D:/Google Drive/Research/Laser Scanner/Control/Data/3-24-20/2/002/rgb_Layer82.png')
# img_cv = cv2.resize(img_cv,(124,124))
# img_cv = img_cv.reshape(1,124,124,3)
#
# model_load.predict(data_array)

# def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
#     lb = LabelBinarizer()
#     lb.fit(y_test)
#     y_test = lb.transform(y_test)
#     y_pred = lb.transform(y_pred)
#
#     for (idx, c_label) in enumerate(3): # all_labels: no of the labels
#         fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
#         c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
#     c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
#     return roc_auc_score(y_test, y_pred, average=average)
#
# # calling
# valid_generator.reset() # resetting generator
# y_pred = model.predict_generator(valid_generator, verbose = True)
# y_pred = np.argmax(y_pred, axis=1)
# multiclass_roc_auc_score(valid_generator.classes, y_pred)

# Save cnn model
# model.save('cnn_model/my_model_rgb_5_2.h5')
# json_string = model.to_json()
# with open('cnn_model/model_rgb_5_2.json', 'w') as outfile:
#     json.dump(json_string, outfile)

# model_load = models.load_model('cnn_model/my_model_rgb_4_26.h5')
# # data_test = Image.open(
# #     'D:/Google Drive/Research/Laser Scanner/Control/Data/3-24-20/total/CNN_3/rgb/Under ++/78.png')
# data_test = Image.open(
#     'D:/Google Drive/Research/Laser Scanner/Control/Data/3-24-20/total/raw_4/6/rgb_Layer4.png')
# data_test = data_test.crop((559,149,1360,930)) # [149:930, 559:1360, :]
# plt.imshow(data_test)
# data_test = data_test.resize([124, 124])
# data_array = np.array(data_test).reshape(1, 124, 124, 3)
# predict = model_load.predict(data_array)
# clss = int(np.where(predict == np.max(predict))[1] + 1)
# print(clss)
