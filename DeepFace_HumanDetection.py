from dataclasses import dataclass, field
from deepface import DeepFace
import cv2
from time import sleep
from retinaface import RetinaFace

@dataclass
class DeepFace_HumanDetection:
    _width_crop: int = 30
    _height_crop: int = 80
    _face_cascade = None

    def __post_init__(self) -> None:
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def part_normalize(self, image):
        image = cv2.GaussianBlur(image, (1,1), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    def detectByImage(self, image):
        # norm_image = self.part_normalize(image)

        # faces = self._face_cascade.detectMultiScale(norm_image, 15, 3)

        # cv2.imshow('test', image)
        # for (x, y, w, h) in faces:
        #     print("DA")
        #     try:
        #         crop_img = image[y:y+h, x:x+w]
        #         cv2.imshow('test', crop_img)
        #     except:
        #         print("No face detected :(")
            
        #     sleep(10)
        
        # k = cv2.waitKey(30) & 0xff
        # sleep(10)

        # cv2.destroyAllWindows()

        resp = RetinaFace.detect_faces("img.png")

        faces = RetinaFace.extract_faces(img_path = "img.png", align = True)

        for face in faces:
            cv2.imshow('q', face)
            sleep(10)
            k = cv2.waitKey(30) & 0xff
        
        cv2.destroyAllWindows()

