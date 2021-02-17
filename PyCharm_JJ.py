##region imports
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import time

import keras as k
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix
##endregion

##region # tic toc
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
## endregion

## region data importing
from random import shuffle


def create_dataset(img_folder, IMG_HEIGHT=30, IMG_WIDTH=30):
    tic()
    i = 1
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        print(dir1)
        files = os.listdir(os.path.join(img_folder, dir1))
        shuffle(files)
        i = 0
        for file in files:
            if file.endswith('.png'):
                if (i % 5000 == 0):
                    print(i)
                    toc()
                    tic()
                # if (i == 2000):
                # break
                i = i + 1
                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255
                img_data_array.append(image)
                class_name.append(dir1)
    return img_data_array, class_name


# extract the image array and class name
img_folder = r'C:\Users\steve\Documents\GitHub\Stevens_AM_lab\Test_data_labeled\labeled'
img_data, class_name = create_dataset(img_folder)
toc()

## endregion



##region # OneHotEncode
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = class_name
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[1001, :])])
print(inverted)

## endregion


##region test data encoding
img_data=np.array(img_data)
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[999, :])])
print(inverted)
## endregion

##region samenet

class Same_net(tf.keras.Model):
    def __init__(self,
                 filters_1x1=16,
                 filters_3x3_reduce=16,
                 filters_3x3=16,
                 filters_5x5_reduce=16,
                 filters_5x5=16,
                 filters_pool_proj=16,
                 name='inception_3a',
                 inp_shapes=(30, 30, 3)):
        super(Same_net, self).__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same')

        self.conv3x3_reduce = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same')
        self.conv3x3 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same')

        self.conv5x5_reduce = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same')
        self.conv5x5 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same')

        self.convpool = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same')

        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')
        self.Concat = tf.keras.layers.Concatenate()

    def call(self, inputs):
        b1 = self.conv1x1(inputs)
        b1_act = self.act(b1)

        b2 = self.conv3x3_reduce(inputs)
        b2_act = self.act(b2)
        b2_2 = self.conv3x3(b2_act)
        b2_2_act = self.act(b2_2)

        b3 = self.conv5x5_reduce(inputs)
        b3_act = self.act(b3)
        b3_2 = self.conv5x5(b3_act)
        b3_2_act = self.act(b3_2)

        b4 = self.max_pool(inputs)
        b4_2 = self.convpool(b4)
        b4_2_act = self.act(b4_2)

        output = self.Concat([b1_act, b2_2_act, b3_2_act, b4_2_act])
        return output
## endregion

##region Encod_2
class Encod_2(tf.keras.Model):
    def __init__(self,filters_1x1=16):
        super(Encod_2, self).__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(filters_1x1, (3, 3), padding='same')
        self.conv1x1_2 = tf.keras.layers.Conv2D(filters_1x1, (3, 3), padding='same')
        self.max_pool2 = tf.keras.layers.MaxPool2D((2,2),strides=(2, 2), padding='same')
        self.act = tf.keras.layers.Activation('relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.norm = tf.keras.layers.BatchNormalization()
        self.norm2 = tf.keras.layers.BatchNormalization()
    def call(self, inputs):
        b1 = self.conv1x1(inputs)
        b1_norm = self.norm(b1)
        b1_act = self.act(b1_norm)
        b1_droped = self.dropout(b1_act)
        b2 = self.conv1x1_2(b1_droped)
        b2_norm = self.norm2(b2)
        b2_act = self.act(b2_norm)
        b3 =   self.max_pool2(b2_act)
        return b3
## endregion


##region Encod_comp
class Encod_comp(tf.keras.Model):
    def __init__(self, filters_1x1=16):
        super(Encod_comp, self).__init__()
        self.b1 = Same_net()
        self.b12 = Same_net()
        self.b13 = Same_net()

        self.b2 = Encod_2()
        self.b22 = Encod_2()
        self.b23 = Encod_2()

    def call(self, inputs):
        l1 = self.b1(inputs)
        l2 = self.b2(l1)
        l3 = self.b12(l2)
        l4 = self.b22(l3)
        l5 = self.b13(l4)
        # l6 = self.b23(l5)

        return l5
## endregion


##region Class_out
class Class_out(tf.keras.Model):
    def __init__(self, filters_1x1=8):
        super(Class_out, self).__init__()
        self.Flat = tf.keras.layers.Flatten()
        self.D1 = tf.keras.layers.Dense(150, activation='relu')
        self.D2 = tf.keras.layers.Dense(40, activation='relu')
        self.D3 = tf.keras.layers.Dense(12, activation='relu')
        self.D4 = tf.keras.layers.Dense(4, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(0.2)
        # self.D5 = tf.keras.layers.Softmax()
        # self.D0 = Encod_comp()

    def call(self, inputs):
        # l0 = self.D0(inputs)
        # l1 = self.Flat(l0)

        l1 = self.Flat(inputs)
        l1_droped =self.dropout(l1)
        l2 = self.D1(l1_droped)
        l2_droped =self.dropout(l2)
        l3 = self.D2(l2_droped)
        #l3_droped =self.dropout(l3)
        l4 = self.D3(l3)
        l5 = self.D4(l4)
        # l6 = self.D5(l5)

        return l5
## endregion


##region Decoder
class Decoder(tf.keras.Model):
    def __init__(self,filters_1x1=16,same_pad = 0,up_sample=0):
        super(Decoder, self).__init__()
        if (same_pad == 1):
          self.conv1x1 = tf.keras.layers.Conv2DTranspose(filters_1x1, (3, 3), padding='same')
          self.conv1x1_2 = tf.keras.layers.Conv2DTranspose(filters_1x1, (3, 3), padding='same')
        elif (same_pad == 0):
          self.conv1x1 = tf.keras.layers.Conv2DTranspose(filters_1x1, (3, 3), padding='valid')
          self.conv1x1_2 = tf.keras.layers.Conv2DTranspose(filters_1x1, (3, 3), padding='valid')
        else:
           self.conv1x1 = tf.keras.layers.Conv2D(filters_1x1, (3, 3))
           self.conv1x1_2 = tf.keras.layers.Conv2DTranspose(filters_1x1, (3, 3), padding='same')
        if up_sample :
          self.UpSampling2D = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        else:
          self.UpSampling2D = tf.keras.layers.UpSampling2D((1, 1), interpolation='bilinear')
        self.act = tf.keras.layers.Activation('sigmoid')

        self.conv_reduce = tf.keras.layers.Conv2D(filters_1x1, (3, 3))

    def call(self, inputs):
        b1 = self.conv1x1(inputs)
        b1_act = self.act(b1)
        b2 = self.conv1x1_2(b1_act)
        b2_act = self.act(b2)
        b3 =   self.UpSampling2D(b2_act)
        return b3
## endregion


##region Decod_comp

class Decod_comp(tf.keras.Model):
    def __init__(self, filters_1x1=8):
        super(Decod_comp, self).__init__()
        self.b1 = Same_net()
        self.b12 = Same_net()
        self.b13 = Same_net()
        self.b14 = Same_net()

        self.b2 = Decoder(same_pad=1, up_sample=1)
        self.b22 = Decoder(same_pad=1, up_sample=1)
        self.b23 = Decoder(same_pad=1, up_sample=0)
        self.b24 = Decoder(filters_1x1=3, same_pad=2, up_sample=0)

    def call(self, inputs):
        l1 = self.b1(inputs)
        l2 = self.b2(l1)
        l3 = self.b12(l2)
        l4 = self.b22(l3)
        l5 = self.b13(l4)
        l6 = self.b23(l5)
        l7 = self.b14(l6)
        l8 = self.b24(l7)

        return l8
## endregion

##regionModel_comp
class Model_comp(tf.keras.Model):
    def __init__(self,filters_1x1=6):
        super(Model_comp, self).__init__()
        self.encoder = Encod_comp()
        self.decoder = Decod_comp()
        self.classifier = Class_out()
    def call(self, inputs):
        l1 = self.encoder(inputs)
        l2 = self.decoder(l1)
        l3 = self.classifier(l1)
        return [l2,l3]

## endregion


checkpoint_filepath = './tmp'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_output_2_accuracy',
    mode='max',
    save_freq='epoch',
    save_best_only=True)

##region training
model_1 = Model_comp()

#model_1.load('2_3_2021_6am.tf')
#model_1.load_weights('2_3_2021_6am_weights.tf')
#np.load('img_data',img_data)
#np.load('onehot_encoded',onehot_encoded)


model_1.compile(optimizer='adam', loss=['mean_squared_error' , 'categorical_crossentropy'], loss_weights=[2,1],metrics=['accuracy'])
#model_1.compile(loss="mean_squared_error", optimizer='RMSprop')
history = model_1.fit(x=img_data,
    y=[img_data, onehot_encoded],
    batch_size=32,
    epochs=20,
    verbose=1,
    callbacks=[model_checkpoint_callback],
    validation_split=0.2,
    shuffle=True)


## endregion


## region Saving
#model_1.save('2_3_2021_6am.tf')
model_1.save_weights('2_3_2021_6am_weights.tf')
#np.save('img_data',img_data)
#np.save('onehot_encoded',onehot_encoded)
##endregion
model_1.predict


plt.figure()
plt.plot(history.history['output_2_accuracy'], label='output_2_accuracy')
plt.plot(history.history['val_output_2_accuracy'], label='val_output_2_accuracy')
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
Y_pred = model_1.predict(img_data)[1]
y_pred = np.argmax(Y_pred, axis=1)
y_true = np.argmax(onehot_encoded, axis=1)
print('Confusion Matrix')
cm1 = confusion_matrix(y_true = y_true, y_pred=y_pred)
print(cm1)
print('Classification Report')
target_names = label_encoder.classes_
print(classification_report(y_true = y_true, y_pred=y_pred, target_names=target_names))

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

