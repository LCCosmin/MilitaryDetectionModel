from dataclasses import dataclass, field
from tabnanny import verbose
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

@dataclass(kw_only=True, frozen=True)
class MilitaryVehicleModel:
    _width_crop: int = 20
    _height_crop: int = 40
    _epochs_no: int = 128
    _batch_size: int = 16
    _checkpoint_path: str = field(init=False)
    _training_folder: str = field(init=False)

    def __post_init__(self) -> None:
        self._training_folder = "./training_dataset_vehicle/"
        self._checkpoint_path = "./models/vehicle/cp.ckpt"
    
    def normalize(self, img):
        img = cv2.resize(img, (self._width_crop, self._height_crop), interpolation = cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (1,1), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #img = img / 255
        #img = cv2.Canny(img, 10, 255)

        return img

    def create_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(self._width_crop * self._height_crop,)),
            keras.layers.Dense(32, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(128, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1024, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            
            tf.keras.layers.Dense(2, activation = 'softmax')
            ])
        
        model.compile(optimizer = 'adam', 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics = ['accuracy'])

        return model

    def gather_training_data(self):
        x_load_images = []
        y_load_images = []
        for filename in os.listdir(self.training_folder):
            #Read one image from folder
            img = cv2.imread(os.path.join(self.training_folder,filename))
            if img is not None:
                #Normalize all images to the same dimensions
                img = self.normalize(img)

                x_load_images.append(img)

                if "military" in filename:
                    y_load_images.append(1)
                else:
                    y_load_images.append(0)

        return (x_load_images, y_load_images)

    def train_and_save(self) -> None:
        #Get the training data
        x_load_images = []
        y_load_images = []
        
        x_load_images, y_load_images = self.gather_training_data()

        x_vehicle = np.array(x_load_images, dtype=(float))
        y_vehicle = np.array(y_load_images, dtype=(int))

        x_train, x_test, y_train, y_test = train_test_split(x_vehicle, y_vehicle, 
                                                            test_size = 0.2, shuffle=(True))

        tf.keras.utils.normalize(x_train, order=2)
        tf.keras.utils.normalize(x_test, order=2)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(self._checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose = 1)

        model = self.create_model()

        model.fit(x_train, y_train, epochs = self._epochs_no, batch_size = self._batch_size, callbacks = [cp_callback])

        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=2)

        print("Accuracy : " + str(accuracy))

    def evaluate_frames(self, vehicle_list) -> None:
        model = self.create_model()
        model.load_weights(self._checkpoint_path)

        for person in vehicle_list:
            person = self.normalize(person)

        predictions = model.predict(vehicle_list)

        print(predictions)
