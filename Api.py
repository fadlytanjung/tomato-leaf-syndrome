from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras import optimizers
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import cv2 as cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,os
# %matplotlib inline

class Api:

    def __init__(self):
        self.data = []

    def read_csv(self, path):
        return pd.read_csv(path)
    
    def read_img(self, path):
        img = cv2.imread(path)
        img = cv2.medianBlur(img,5)
        return img
    
    def save_img(self, img, fileName):

        # plt.imshow(img)
        print(img,fileName)
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join('static/tempData/' , fileName), rgbImg)
        # cv2.waitKey(0)
        return 'save immage' + fileName + 'successful'

    def scale_img(self, img,scale=10, autoScale=True, width=302, height=403):
        scale_percent = scale # percent of original size
        widthScale = int(img.shape[1] * scale_percent / 100)
        heightScale = int(img.shape[0] * scale_percent / 100)
        dim = (widthScale, heightScale)
        customDIm = (width,height)
        # resize image
        resized = cv2.resize(img, dim if autoScale else customDIm, cv2.COLOR_BGR2RGB)
        return resized
    
    def grayscale(self, img):
        img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    def rgb_hsv(self, img):
        img_rgb_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img_rgb_hsv

    def threshold(self, img):
        ret, img_treshold = cv2.threshold(img,127,255, cv2.THRESH_BINARY_INV)
        return img_treshold

    def modelCNN(self):
        model = Sequential()
        #1-Convolution
        model.add(Conv2D(32, (3,3), strides=(1,1), input_shape = (300,300,3), activation = 'relu'))
        #2-Pooling
        model.add(MaxPooling2D(pool_size = (2,2)))
        #add second conv
        model.add(Conv2D(32, (3,3), strides=(1,1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        #add third conv
        model.add(Conv2D(32, (3,3), strides=(1,1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        #add third conv
        model.add(Conv2D(64, (3,3), strides=(1,1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        #add third conv
        model.add(Conv2D(64, (3,3), strides=(1,1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        #3-Flattening
        model.add(Flatten())
        #4-Full Connection
        model.add(Dense(activation = 'relu', units = 900))
        model.add(Dense(activation = 'relu', units = 90))
        model.add(Dense(activation = 'softmax', units = 3))


        #Compiling CNN
        rmsprop = optimizers.RMSprop(learning_rate=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer = rmsprop, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.summary()

        return model

    def train(self,model, train_path,validation_path):

        batch_size = 16

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

        # this is a generator that will read pictures found in subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_generator = train_datagen.flow_from_directory(
                train_path,  # this is the target directory
                target_size=(300,300),  # all images will be resized to 300x300
                batch_size=batch_size,
                class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
                validation_path,
                target_size=(300,300),
                batch_size=batch_size,
                class_mode='categorical')

        history_callback = model.fit_generator(
        train_generator,
        steps_per_epoch=1533 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=305 // batch_size)

        model.save('data/model.h5')  # always save your weights after training or during training


        loss_history = history_callback.history["loss"]
        accuracy_history = history_callback.history["accuracy"]
        val_loss_history = history_callback.history["val_loss"]
        val_accuracy_history = history_callback.history["val_accuracy"]

        numpy_loss_history = np.array(loss_history)
        np.savetxt("static/assets/loss_history.txt", numpy_loss_history, delimiter=",")
        #return numpy_loss_history
        numpy_accuracy_history = np.array(accuracy_history)
        np.savetxt("static/assets/acc_history.txt", numpy_accuracy_history, delimiter=",")

        numpy_val_loss_history = np.array(val_loss_history)
        np.savetxt("static/assets/val_loss_history.txt", numpy_val_loss_history, delimiter=",")
        #return numpy_loss_history
        numpy_val_accuracy_history = np.array(val_accuracy_history)
        np.savetxt("static/assets/val_acc_history.txt", numpy_val_accuracy_history, delimiter=",")

        self.save_graph(history=history_callback)

        return 'Train Successful'

    def save_graph(self, history):
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('static/assets/accuracy.png')
        plt.close()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('static/assets/loss.png')
        plt.close()
        
        return "Save Successful"

    def loadModel(self, path):
        model_loaded = load_model(path)
        rmsprop = optimizers.RMSprop(learning_rate=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
        model_loaded.compile(loss='categorical_crossentropy',
                    optimizer=rmsprop,
                    metrics=['accuracy'])
        model_loaded.summary()
        print('Load Model Successfull')
        return model_loaded
    
    def load_image(img_path, show=False):
        img = image.load_img(img_path, target_size=(300, 300))
        img = (np.asarray(img))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        return img_tensor

    def predict(self, model, pathImg):
        print('predicting')
        img = image.load_img(pathImg, target_size=(300, 300))
        img = (np.asarray(img))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        predImg = model.predict(img_tensor)
        print('value sermentation: ',np.argmax(model.predict(img_tensor), axis=-1))
        print('result',predImg)
        
        classes_pred=np.argmax(predImg,axis=1)
        result = ['healthy-leaves', 'liriomyza-leafspot','septoria-leafspot']
        
        return result[classes_pred[0]]