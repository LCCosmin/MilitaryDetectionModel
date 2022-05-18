from dataclasses import dataclass, field
import cv2
import imutils
import numpy as np

@dataclass
class HOG_HumanDetection:
    _HOGCV = None

    def __post_init__(self) -> None:
        self._HOGCV = cv2.HOGDescriptor()
        self._HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        bounding_box_cordinates, weights =  self.HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
        
        cropped_people = []

        for x,y,w,h in bounding_box_cordinates:
            cropped_people.append(frame[x:y, (x+w):(y+h)])
    
        return cropped_people
    
    def detectByImage(self, frame_list):
        cropped_people = []

        for frame in frame_list:
            cv2.imshow('q',frame)
            image = imutils.resize(frame, width = min(800, frame.shape[1])) 
            
            result_images = self.detect(image)
            for people in result_images:
                cropped_people.append(people)
    
        #print(cropped_people)
        return cropped_people