from HOG_HumanDetection import HOG_HumanDetection
from VideoReader import VideoCapture
from VehicleDetection import VehicleDetection
from MilitaryPersonnelModel import MilitaryPersonnelModel
from MilitaryVehicleModel import MilitaryVehicleModel
from ImageReader import ImageReader
from DeepFace_HumanDetection import DeepFace_HumanDetection

def main():
    #VideoRead = VideoCapture("video.mp4", 1)
    ImageRead = ImageReader("img.png")

    # HOG_HumanDetector = HOG_HumanDetection()
    # VehicleDetector = VehicleDetection()
    DF_HumanDetector = DeepFace_HumanDetection()

    #PersonnelModel = MilitaryPersonnelModel()
    # VehicleModel = MilitaryVehicleModel()

    # print("Opening Video capture module ... ")
    # frame_list = VideoRead.read_by_frame()
    print("Opening Image capture module ...")
    img = ImageRead.read_by_image()

    # print("Using Human Detector module to detect people from a video ...")
    # people_list = HumanDetector.detectByVideo(frame_list)
    print("Using Human Detector module to detect people from an image ...")
    #people_list = HumanDetector.detectByImage(img)
    people_list = DF_HumanDetector.detectByImage(img)

    # Broken
    # print("Using Vehicle Detector module to detect vehicles ...")
    # vehicle_list = VehicleDetector.detectByImage(frame_list)

    #PersonnelModel.train_and_save()
    # VehicleModel.train_and_save()

    # print("Using ML Model to determine if a person is a military personnel or not from a video ...")
    # PersonnelModel.evaluate_frames(people_list)
    # VehicleModel.evaluate_frames(vehicle_list)
    #print("Using ML Model to determine if a person is a military personnel or not from an image ...")
    #PersonnelModel.evaluate_image(people_list)
    
if __name__ == "__main__":
    main()
else:
    print("Error when importing")