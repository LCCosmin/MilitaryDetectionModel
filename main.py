from face_detector.HOG_HumanDetection import HOG_HumanDetection
from face_detector.RetinaFace_HumanDetection import RetinaFace_HumanDetection
from utils.VideoReader import VideoCapture
from utils.ImageReader import ImageReader
from ml_model.MilitaryPersonnelModel import MilitaryPersonnelModel

def main():
    VideoRead = VideoCapture("hacksaw-ridge.mkv", 120)
    # ImageRead = ImageReader("")

    # # HOG_HumanDetector = HOG_HumanDetection() <- Low efficiency
    RF_HumanDetector = RetinaFace_HumanDetection()

    PersonnelModel = MilitaryPersonnelModel()

    print("Opening Video capture module ... ")
    frame_list = VideoRead.read_by_frame()
    # print("Opening Image capture module ...")
    # img = ImageRead.read_by_image()

    print("Using Human Detector module to detect people from a video ...")
    people_list = RF_HumanDetector.detect_by_video(frame_list)
    # print("Using Human Detector module to detect people from an image ...")
    # people_list = HOG_HumanDetector.detectByImage(img) <- Low efficiency
    # people_list = RF_HumanDetector.detect_by_image(img)

    # PersonnelModel.train_and_save()

    print("Using ML Model to determine if a person is a military personnel or not from a video ...")
    PersonnelModel.evaluate_frames(people_list)
    # print("Using ML Model to determine if a person is a military personnel or not from an image ...")
    # PersonnelModel.evaluate_image(people_list)
    

if __name__ == "__main__":
    main()
else:
    print("Error when importing")