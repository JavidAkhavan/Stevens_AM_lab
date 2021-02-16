'''
Created on 04/06/2020

Build a CNN model to classification the image from point cloud

'''
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from PIL import Image
import glob
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from imutils import paths


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


checkpoint_filepath = 'D:\\local_research\\conference_2\\tmp\\'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# %% data generator
'''
 Use keras to import all the training data. Each folder represent one category.

datagen = keras.preprocessing.image.ImageDataGenerator(
    # rotation_range=90,  # Degree range for random rotations.
    # width_shift_range=0, # fraction of total width
    # height_shift_range=0, # fraction of total height
    # shear_range=0, # ramdom change
    # zoom_range=[1.0, 1.5],  # ramdom zoom
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
    target_size=(30, 30),
    color_mode='rgb',
    batch_size=64,
    class_mode='categorical',
    subset='training',

)
# 'D:/Google Drive/Research/Laser Scanner/Control/Data/3-24-20/total/CNN_6/rgb/'
validation_generator = datagen.flow_from_directory(
    'Test_data_labeled/labeled/',
    #    target_size=(411, 411),
    target_size=(30, 30),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

def custom_sparse_categorical_accuracy(y_true, y_pred):
    flatten_y_true = K.cast( K.reshape(y_true,(-1,1) ), K.floatx())
    flatten_y_pred = K.cast(K.reshape(y_pred, (-1, y_pred.shape[-1])), K.floatx())
    y_pred_labels = K.cast(K.argmax(flatten_y_pred, axis=-1), K.floatx())
    return K.cast(K.equal(flatten_y_true,y_pred_labels), K.floatx())
'''


# Load images
def load_data(data_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images')
    for fld in classes:  # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(data_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
            # if i > 0 and ((i + 1) % 200 == 0 or i == len(imagePaths) - 1):
            #     print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


classes = ['Empty', 'OK', 'Over', 'Under']
data_path = 'Test_data_labeled/labeled/'
img_size = 30
rawImages, labels, ids, cls = load_data(data_path, img_size, classes)

# print("[INFO] handling images...")
# imagePaths = list(paths.list_images('Test_data_labeled/labeled/'))
# rawImages = np.zeros([len(imagePaths), 30, 30, 3])
# labels = []
# # loop over the input images
# for (i, imagePath) in enumerate(imagePaths):
#     # load the image and extract the class label
#     # our images were named as labels.image_number.format
#     image = cv2.imread(imagePath)
#     if image is None:
#         # print('skip')
#         continue
#     if len(image) != 30:
#         image = cv2.resize(image, (30, 30))
#     # get the labels from the name of the images by extract the string before "."
#     label = imagePath.split(os.path.sep)[-2]
#     # label = imagePath.split(os.path.sep)[-1].split(".")[0].split('_')[:-1]
#     # label = "_".join(label)
#     rawImages[i, :, :, :] = image
#     labels.append(label)
#
#     # show an update every 200 images until the last image
#     if i > 0 and ((i + 1) % 200 == 0 or i == len(imagePaths) - 1):
#         print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

X_train, X_test, y_train, y_test = train_test_split(rawImages, labels, test_size=0.2, random_state=42)

#################################################################
regu = regularizers.l2(0.0002)
# %% Mine CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.002), input_shape=(30, 30, 3)))
model.add(layers.BatchNormalization())  # scale=False, center=False
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.002)))
model.add(layers.BatchNormalization())  # scale=False, center=False
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.002)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.0002)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
#
# model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.002)))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
# # model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.0002)))
# # model.add(layers.BatchNormalization())
# # model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.3))

# model.add(layers.Conv2D(24, (5, 5), activation='relu'))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization(scale=False, center=False))
# model.add(layers.MaxPooling2D((2, 2)))
# # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# # model.add(layers.Dropout(0.25))


model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.002)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.002)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())

# model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))

model.summary()
# load model weight
model.load_weights('checkpoint/model_weights_2_15_21')

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',  # sparse_categorical_crossentropy
              metrics=['accuracy']  # ,f1_m,precision_m, recall_m, 'accuracy'custom_sparse_categorical_accuracy
              )  # 'adam',  # optimizer = keras.optimizers.RMSprop(lr=1e-4)
# loss='categorical_crossentropy'

history = model.fit(
    x=X_train,  # train_generator
    y=y_train,
    # steps_per_epoch=30,
    epochs=100,
    validation_data=(X_test, y_test),  # validation_generator
    validation_steps=90,
    callbacks=[model_checkpoint_callback]
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
Y_pred = model.predict(X_test)  # predict_generator(validation_generator, 407)
y_pred = np.argmax(Y_pred, axis=1)
y_test_cm = np.argmax(y_test, axis=1)
print('Confusion Matrix')
cm1 = confusion_matrix(y_test_cm, y_pred)
print(cm1)
print('Classification Report')
target_names = ['Empty', 'OK', 'Over', 'Under']
print(classification_report(y_test_cm, y_pred, target_names=target_names))  #

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

# Save the weights
# model.save_weights('checkpoint/model_weights_2_15_21')

# cmm = np.array([
#     [1403,49,45,0],
#     [422,563,32,0],
#     [75,16,	886,189],
#     [0,0,222,534]
# ])
#
# cm_df = pd.DataFrame(cmm,
#                      index = ['Normal','Over extrusion','Under extrusion', 'Severe under extrusion'],
#                      columns = ['Normal','Over extrusion','Under extrusion', 'Severe under extrusion'])
#
# plt.figure(figsize=(5.5,4))
# sns.heatmap(cm_df, annot=True)
# plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(0.91))
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()

# Save cnn model
# model.save('cnn_model/model_weights_7_21.h5')


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

# # # Write to excel
# import xlsxwriter
# workbook = xlsxwriter.Workbook('D:/Google Drive/Research/Journal_4/CNN_output_7_21(2).xlsx')
# worksheet = workbook.add_worksheet()
# worksheet.write(0, 0, 'accuracy')
# worksheet.write(0, 1, 'val_accuracy')
# worksheet.write(0, 2, 'loss')
# worksheet.write(0, 3, 'val_loss')
#
# for i in range(len(history.history['accuracy'])):
#     worksheet.write(i+1, 0, history.history['accuracy'][i])
#     worksheet.write(i+1, 1, history.history['val_accuracy'][i])
#     worksheet.write(i+1, 2, history.history['loss'][i])
#     worksheet.write(i+1, 3, history.history['val_loss'][i])
# workbook.close()
