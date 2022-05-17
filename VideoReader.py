from dataclasses import dataclass, field
import cv2

@dataclass
class VideoCapture:
    _video_path: str
    _frames_no: int

    def update_video_path(self, path) -> None:
        self._video_path = path

    def update_frames_no(self, no) -> None:
        self._frames_no = no

    def read_by_frame(self):
        video_capture = cv2.VideoCapture(self._video_path)
        success, frame = video_capture.read()
        current_frame = 1
        frame_list = []

        while success:
            if current_frame % self._frames_no == 0:
                frame_list.append(frame)
            
            current_frame += 1
            success, frame = video_capture.read()

        print(frame_list)
        return frame_list