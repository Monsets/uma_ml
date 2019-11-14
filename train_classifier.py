import cv2
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


class ClassifierTrainer():
    def __init__(self, model, path_to_images='images', saving_weights_path='weights.hdf5',
                 batch_size=64, seed=3, num_epochs=100, train_size=0.85, learning_rate=1e-3):
        self.__seed = seed
        self.__num_epochs = num_epochs
        self.__path_to_images = path_to_images
        self.__batch_size = batch_size
        self.__train_size = train_size
        self.__learning_rate = learning_rate
        self.__img_size = model.input_shape[1]
        self.__model = model
        self.__saving_weights_path = saving_weights_path

        self.__read_data()
        self.__get_train_test_split()
        self.__build_generators()
        self.__build_training_model()

    def __read_data(self):
        self.__data = pd.read_csv('images_labelling.csv')
        self.__data['boxid'] = self.__data['boxid'].apply(lambda t: str(t) + '.png')
        self.__data['label'] = self.__data['label'].astype('str')

    def __get_train_test_split(self):
        self.__train_set = self.__data.sample(frac = self.__train_size, random_state = self.__seed)
        self.__test_set = self.__data.drop(self.__train_set.index)

    def __preprocess_image(self, img):
        return cv2.resize(img, (self.__img_size, self.__img_size))

    def __build_generators(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                         horizontal_flip=True,
                                           rotation_range=30,
                                           zoom_range=0.2,
                                           shear_range=0.1,
                                           fill_mode='nearest',
                                           preprocessing_function=self.__preprocess_image)

        test_datagen = ImageDataGenerator(rescale=1. / 255,
                                          fill_mode='nearest',
                                          shear_range=0.1,
                                          preprocessing_function=self.__preprocess_image)

        self.__train_generator = train_datagen.flow_from_dataframe(dataframe=self.__train_set,
                                                                 directory=self.__path_to_images,
                                                                 x_col="boxid",
                                                                 y_col="label",
                                                                 batch_size=self.__batch_size,
                                                                 target_size=(self.__img_size, self.__img_size),
                                                                 class_mode='categorical',
                                                                 shaffle=True,
                                                                 seed=self.__seed,
                                                                 )
        self.__test_generator = test_datagen.flow_from_dataframe(dataframe=self.__test_set,
                                                               directory=self.__path_to_images,
                                                               x_col="boxid",
                                                               y_col="label",
                                                               batch_size=self.__batch_size,
                                                               target_size=(self.__img_size, self.__img_size),
                                                               class_mode='categorical',
                                                               shaffle=True,
                                                               seed=self.seed
                                                               )
        self.__num_train_steps = self.__train_generator.n // self.__train_generator.batch_size
        self.__num_test_steps = self.__test_generator.n // self.__test_generator.batch_size

    def __build_training_model(self):
        optimizer = SGD(lr=self.__learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.__checkpoints = ModelCheckpoint(filepath=self.__saving_weights_path, monitor='val_loss',
                                           save_weights_only=True, save_best_only=True)
        self.__model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    def fit(self):
        self.__model.fit_generator(generator=self.__train_generator,
                                 steps_per_epoch=self.__num_train_steps,
                                 validation_data=self.__test_generator,
                                 validation_steps=self.__num_test_steps,
                                 epochs=self.__num_epochs,
                                 callbacks=[self.__checkpoints]
                                 )
