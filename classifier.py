import cv2
import numpy as np
from efficientnet.keras import EfficientNetB0
from keras.layers import Dropout, Dense
from keras.layers import GlobalAveragePooling2D, Input
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator

from train_classifier import ClassifierTrainer
from utils import label_accordance_table


class Classifier():
    def __init__(self, path_to_weights='weights.hdf5', n_classes=25):
        self.path_to_weights = path_to_weights
        self.img_size = 224
        self.n_channel = 3
        self.n_classes = n_classes

        self.__build_model()

    def __preprocess_image(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size))
        return np.array(img).reshape((-1, self.img_size, self.img_size, 3))

    def __build_preprocess_generator(self):
        self.pred_datagen = ImageDataGenerator(rescale=1. / 255,
                                               fill_mode='nearest',
                                               shear_range=0.1,
                                               preprocessing_function=self.__preprocess_image)

    def __build_model(self):
        input_tensor = Input(shape=(self.img_size, self.img_size, self.n_channel))
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_tensor)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.3)(x)
        output_layer = Dense(self.n_classes, activation='softmax', name="Output_Layer")(x)
        self.model = Model(input_tensor, output_layer)

        try:
            self.model.load_weights(self.path_to_weights)
        except:
            print("can't load weights! \n training...")
            try:
                ClassifierTrainer(self.model).fit()
            except:
                print("Unable to train model! Exiting...")
                exit(1)
            print("Training has finished! Ready to work.")
        print('Model is ready')
        self.model._make_predict_function()

    def predict(self, img):
        pred_generator = self.pred_datagen.flow(x=img)
        pred = np.argmax(self.model.predict(pred_generator))
        pred = label_accordance_table[pred]
        return pred
