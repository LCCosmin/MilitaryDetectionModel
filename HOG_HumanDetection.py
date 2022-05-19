from dataclasses import dataclass, field
import cv2
import imutils
import numpy as np

@dataclass
class HOG_HumanDetection:
    _width_crop: int = 30
    _height_crop: int = 80
    _HOGCV = None

    def __post_init__(self) -> None:
        self._HOGCV = cv2.HOGDescriptor()
        self._HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        bounding_box_cordinates, weights =  self._HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
        
        cropped_people = []

        for x,y,w,h in bounding_box_cordinates:
            cropped_people.append(frame[x:y, (x+w):(y+h)])
    
        return cropped_people
    
    def detectByVideo(self, frame_list):
        cropped_people = []

        for frame in frame_list:
            result_images = self.detect(frame)
            for people in result_images:
                if people.any():
                    #TO DO, same as in detect by image
                    people = self.normalize(people)
                    people = np.array(cv2.resize(people, (30, 80)), dtype = float).reshape(-1, 30 * 80)
                    cropped_people.append(people)

        return cropped_people

    def detectByImage(self, image):
        cropped_people = []
        result_images = self.detect(image)
        
        for people in result_images:
            if people.any():
                cropped_people.append(people)

        return cropped_people