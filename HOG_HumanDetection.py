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
        bounding_box_cordinates, weights =  self._HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
        
        cropped_people = []

        for x,y,w,h in bounding_box_cordinates:
            # print(x)
            # print(y)
            # print(w)
            # print(h)
            # print("------------------")
            cropped_people.append(frame[x:y, (x+w):(y+h)])
    
        return cropped_people
    
    def detectByImage(self, frame_list):
        cropped_people = []

        for frame in frame_list:
            #cv2.imshow('q', frame)
            # print(frame.ndim)
            # image = cv2.resize(frame, (min(800, frame.shape[1]), min(800, frame.shape[0])), interpolation = cv2.INTER_AREA) 
            
            result_images = self.detect(frame)
            for people in result_images:
                if people.any():
                    people = np.array(cv2.resize(people, (30, 80)), dtype = float).reshape(-1, 30 * 80)
                    cropped_people.append(people)

                    # cv2.imshow('q', people)
                    # key = cv2.waitKey(1)
                    # if key==ord('q'):
                    #     break
    
        #print(cropped_people)
        #cv2.destroyAllWindows()
        return cropped_people