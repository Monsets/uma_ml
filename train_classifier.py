import cv2
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


class ClassifierTrainer():
    def __init__(self, model, path_to_images='images', saving_weights_path='weights.hdf5',
                 batch_size=64, seed=3, num_epochs=100, train_size=0.85, learning_rate=1e-3):
        self.seed = seed
        self.num_epochs = num_epochs
        self.path_to_images = path_to_images
        self.batch_size = batch_size
        self.train_size = train_size
        self.learning_rate = learning_rate
        self.img_size = model.input_shape[1]
        self.model = model
        self.saving_weights_path = saving_weights_path

        self.__read_data()
        self.__get_train_test_split()
        self.__build_generators()
        self.__build_training_model()

    def __read_data(self):
        self.data = pd.read_csv('images_labelling.csv')
        self.data['boxid'] = self.data['boxid'].apply(lambda t: str(t) + '.png')
        self.data['label'] = self.data['label'].astype('str')

    def __get_train_test_split(self):
        self.train_set = self.data.sample(frac=self.train_size, random_state=self.seed)
        self.test_set = self.data.drop(self.train_set.index)

    def __preprocess_image(self, img):
        return cv2.resize(img, (self.img_size, self.img_size))

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

        self.train_generator = train_datagen.flow_from_dataframe(dataframe=self.train_set,
                                                                 directory=self.path_to_images,
                                                                 x_col="boxid",
                                                                 y_col="label",
                                                                 batch_size=self.batch_size,
                                                                 target_size=(self.img_size, self.img_size),
                                                                 class_mode='categorical',
                                                                 shaffle=True,
                                                                 seed=self.seed,
                                                                 )
        self.test_generator = test_datagen.flow_from_dataframe(dataframe=self.test_set,
                                                               directory=self.path_to_images,
                                                               x_col="boxid",
                                                               y_col="label",
                                                               batch_size=self.batch_size,
                                                               target_size=(self.img_size, self.img_size),
                                                               class_mode='categorical',
                                                               shaffle=True,
                                                               seed=self.seed
                                                               )
        self.num_train_steps = self.train_generator.n // self.train_generator.batch_size
        self.num_test_steps = self.test_generator.n // self.test_generator.batch_size

    def __build_training_model(self):
        optimizer = SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.checkpoints = ModelCheckpoint(filepath=self.saving_weights_path, monitor='val_loss',
                                           save_weights_only=True, save_best_only=True)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    def fit(self):
        self.model.fit_generator(generator=self.train_generator,
                                 steps_per_epoch=self.num_train_steps,
                                 validation_data=self.test_generator,
                                 validation_steps=self.num_test_steps,
                                 epochs=self.num_epochs,
                                 callbacks=[self.checkpoints]
                                 )
