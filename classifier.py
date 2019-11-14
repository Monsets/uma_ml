import cv2
import numpy as np
from keras.layers import Dropout, Dense
from keras.layers import GlobalAveragePooling2D, Input
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50

from train_classifier import ClassifierTrainer
from utils import label_accordance_table


class Classifier():
    def __init__(self, path_to_weights='weights.hdf5', n_classes=25):
        self.__path_to_weights = path_to_weights
        self.__img_size = 224
        self.__n_channel = 3
        self.__n_classes = n_classes

        self.__build_model()

    def __preprocess_image(self, img):
        img = cv2.resize(img, (self.__img_size, self.__img_size))
        return np.array(img).reshape((-1, self.__img_size, self.__img_size, 3))

    def __build_preprocess_generator(self):
        self.__pred_datagen = ImageDataGenerator(rescale=1. / 255,
                                               fill_mode='nearest',
                                               shear_range=0.1,
                                               preprocessing_function=self.__preprocess_image)

    def __build_model(self):
        input_tensor = Input(shape=(self.__img_size, self.__img_size, self.__n_channel))
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.3)(x)
        output_layer = Dense(self.__n_classes, activation='softmax', name="Output_Layer")(x)
        self.__model = Model(input_tensor, output_layer)

        try:
            self.__model.load_weights(self.__path_to_weights)
        except:
            print("can't load weights! \n training...")
            try:
                ClassifierTrainer(self.__model).fit()
            except:
                print("Unable to train model! Exiting...")
                exit(1)
            print("Training has finished! Ready to work.")
        print('Model is ready')
        self.__model._make_predict_function()

    def predict(self, img):
        pred_generator = self.__pred_datagen.flow(x=img)
        pred = np.argmax(self.__model.predict(pred_generator))
        pred = label_accordance_table[pred]
        return pred
