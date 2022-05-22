from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from face_detector.RetinaFace_HumanDetection import RetinaFace_HumanDetection


@dataclass(kw_only=True)
class MilitaryPersonnelModel:
    _width_crop: int = 20
    _height_crop: int = 60
    _epochs_no: int = 256
    _batch_size: int = 32
    _checkpoint_path: str = field(init=False)
    _training_folder: str = field(init=False)

    def __post_init__(self) -> None:
        self._training_folder = "./training_dataset_personnel/"
        self._checkpoint_path = "./models/personnel/cp.ckpt"
    
    def normalize(self, img):
        img = cv2.resize(img, (self._width_crop, self._height_crop), interpolation = cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (1,1), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = img / 255
        #img = cv2.Canny(img, 10, 255)

        return img

    def create_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(self._width_crop * self._height_crop,)),
            keras.layers.Dense(400, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(200, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(100, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(50, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(2, activation = 'sigmoid')
            ])
        
        model.compile(optimizer = 'adam', 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics = ['accuracy'])

        return model

    def gather_training_data(self):
        print("Gather all data for training ...")
        x_load_images = []
        y_load_images = []
        for filename in os.listdir(self._training_folder):
            #Read one image from folder
            img = cv2.imread(os.path.join(self._training_folder,filename))
            if img is not None:
                face_detector = RetinaFace_HumanDetection()
                persons = face_detector.detect_by_image(img)

                for person in persons:
                    person = self.normalize(person)
                    x_load_images.append(person)

                    if "military" in filename:
                        y_load_images.append(1)
                    else:
                        y_load_images.append(0)


        print("For training were used:")
        print("{} pictures with non-military people ".format(y_load_images.count(0)))
        print("{} pictures with military people".format(y_load_images.count(1)))
        return (x_load_images, y_load_images)

    def train_and_save(self) -> None:
        #Get the training data
        print("Starting the training protocol ...")
        x_load_images = []
        y_load_images = []

        x_load_images, y_load_images = self.gather_training_data()

        print ("Start training ...")
        x_personnel = np.array(x_load_images, dtype=(float)).reshape(-1, self._width_crop * self._height_crop)
        y_personnel = np.array(y_load_images, dtype=(int))

        x_train, x_test, y_train, y_test = train_test_split(x_personnel, y_personnel, 
                                                            test_size = 0.2, shuffle=(True))

        model = self.create_model()

        history = model.fit(x_train, y_train, epochs = self._epochs_no, batch_size = self._batch_size)
        model.save_weights(self._checkpoint_path)

        loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

        print(model.predict(x_test)[0])

        plt.plot(history.history['loss'])
        plt.show(block=True)
        plt.savefig("./loss.png")

        print("Accuracy : " + str(accuracy))

        print("Training done ..")

    def evaluate_frames(self, people_list) -> None:
        model = self.create_model()
        model.load_weights(self._checkpoint_path).expect_partial()

        n = 1

        for person in people_list:
            copy_image = person
            person = self.normalize(person)
            person = np.array(cv2.resize(person, (self._width_crop, self._height_crop)), dtype = float).reshape(-1, self._width_crop * self._height_crop)
            
            predictions = model.predict(person)
            if predictions[0][0] > 0.5:
                cv2.imwrite("./saves/persons/non-military/image-video{}.jpg".format(n), copy_image)
            else:
                cv2.imwrite("./saves/persons/military/image-video{}.jpg".format(n), copy_image)
            n = n + 1

    def evaluate_image(self, image) -> None:
        model = self.create_model()
        model.load_weights(self._checkpoint_path).expect_partial()

        n = 1

        for person in image:
            copy_image = person
            person = self.normalize(person)
            person = np.array(cv2.resize(person, (self._width_crop, self._height_crop)), dtype = float).reshape(-1, self._width_crop * self._height_crop)
            
            predictions = model.predict(person)
            if predictions[0][0] <= 0.5:
                cv2.imwrite("./saves/persons/non-military/image{}.jpg".format(n), copy_image)
            else:
                cv2.imwrite("./saves/persons/military/image{}.jpg".format(n), copy_image)
            n = n + 1