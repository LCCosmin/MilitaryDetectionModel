from HOG_HumanDetection import HOG_HumanDetection
from VideoReader import VideoCapture
from VehicleDetection import VehicleDetection
from MilitaryPersonnelModel import MilitaryPersonnelModel
from MilitaryVehicleModel import MilitaryVehicleModel

def main():
    VideoRead = VideoCapture("video_2.mp4", 1)
    
    HumanDetector = HOG_HumanDetection()
    VehicleDetector = VehicleDetection()

    PersonnelModel = MilitaryPersonnelModel()
    VehicleModel = MilitaryVehicleModel()

    frame_list = VideoRead.read_by_frame()
    people_list = HumanDetector.detectByImage(frame_list)
    vehicle_list = VehicleDetector.detectByImage(frame_list)

    PersonnelModel.train_and_save()
    VehicleModel.train_and_save()

    PersonnelModel.evaluate_frames(people_list)
    VehicleModel.evaluate_frames(vehicle_list)
    
if __name__ == "__main__":
    main()
else:
    print("Error when importing")