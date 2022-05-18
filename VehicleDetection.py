from dataclasses import dataclass, field
import cv2
import numpy as np

@dataclass(kw_only=True)
class VehicleDetection:
    _min_width_rectangle: int = 80
    _min_height_rectangle: int = 80
    _count_line_position: int = 550
    _algo = None

    def __post_init__(self) -> None:
        self._algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    def center_handle(x,y,w,h):
        x1=int(w/2)
        y1=int(h/2)
        cx=x+x1
        cy=y+y1
        return cx,cy

    def detectByImage(self, frame_list):
        cropped_vehicles = []

        for frame in frame_list:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3,3), 5)
            
        # Applying on each frame
            vid_sub = self._algo.apply(blur)
            dilat = cv2.dilate(vid_sub, np.ones((5,5)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
            dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
            countersahpe, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.line(frame, (25,self._count_line_position),(1200,self._count_line_position),(255,0,0), 3)

            for (i, c) in enumerate(countersahpe):
                (x,y,w,h) = cv2.boundingRect(c)
                val_counter = (w>=self._min_width_rectangle) and (h>= self._min_height_rectangle)
                if not val_counter:
                    continue

                cropped_vehicles.append(frame[x:y, (x+w, y+h)])
        
        #print(cropped_vehicles)
        return cropped_vehicles

