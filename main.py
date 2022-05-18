from HOG_HumanDetection import HOG_HumanDetection
from VideoReader import VideoCapture
from VehicleDetection import VehicleDetection
from MilitaryPersonnelModel import MilitaryPersonnelModel
from MilitaryVehicleModel import MilitaryVehicleModel

def main():
    
    VideoRead = VideoCapture("video.mp4", 1)
    
    HumanDetector = HOG_HumanDetection()
    # VehicleDetector = VehicleDetection()

    PersonnelModel = MilitaryPersonnelModel()
    # VehicleModel = MilitaryVehicleModel()

    print("Opening Video capture module ... ")
    frame_list = VideoRead.read_by_frame()

    print("Using Human Detector module to detect people ...")
    people_list = HumanDetector.detectByImage(frame_list)

    # Broken
    # print("Using Vehicle Detector module to detect vehicles ...")
    # vehicle_list = VehicleDetector.detectByImage(frame_list)

    # PersonnelModel.train_and_save()
    # VehicleModel.train_and_save()

    print("Using ML Model to determine if a person is a military personnel or not ...")
    PersonnelModel.evaluate_frames(people_list)
    # VehicleModel.evaluate_frames(vehicle_list)
    
if __name__ == "__main__":
    main()
else:
    print("Error when importing")