from HOG_HumanDetection import HOG_HumanDetection
from VideoReader import VideoCapture

def main():
    HOGDetection = HOG_HumanDetection()
    VideoRead = VideoCapture("", 1)

    frame_list = VideoRead.read_by_frame()


if __name__ == "__main__":
    main()
else:
    print("Error when importing")