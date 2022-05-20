from HOG_HumanDetection import HOG_HumanDetection
from VideoReader import VideoCapture
from VehicleDetection import VehicleDetection
from MilitaryPersonnelModel import MilitaryPersonnelModel
from MilitaryVehicleModel import MilitaryVehicleModel
from ImageReader import ImageReader
from DeepFace_HumanDetection import DeepFace_HumanDetection

def main():
    # VideoRead = VideoCapture("video.mp4", 10)
    # Images available for test:
    # Non-Military
    # img-test-non-military.jpg
    # img-test-non-military.jpeg
    # img-test-non-military.png
    # img-test-non-military.webp
    # Military
    # img-test-military.webp <- fails hard, suspecting picture format
    # img-test-military.jpg
    # img-test-military.jpeg <- Hardcore, fails hard
    # Mixed
    # img-test-mixed.jpg <- fails super hard
    # img-test-mixed.jpeg
    ImageRead = ImageReader("img-test-mixed.jpeg")

    # # HOG_HumanDetector = HOG_HumanDetection() <- Low efficiency
    # # VehicleDetector = VehicleDetection() <- TO DO find another model, this one is based on movement not shape
    DF_HumanDetector = DeepFace_HumanDetection()

    PersonnelModel = MilitaryPersonnelModel()
    # # VehicleModel = MilitaryVehicleModel() 

    # print("Opening Video capture module ... ")
    # frame_list = VideoRead.read_by_frame()
    print("Opening Image capture module ...")
    img = ImageRead.read_by_image()

    # print("Using Human Detector module to detect people from a video ...")
    # people_list = DF_HumanDetector.detectByVideo(frame_list)
    print("Using Human Detector module to detect people from an image ...")
    # # people_list = HOG_HumanDetector.detectByImage(img) <- Low efficiency
    people_list = DF_HumanDetector.detectByImage(img)

    # # print("Using Vehicle Detector module to detect vehicles ...")
    # # vehicle_list = VehicleDetector.detectByImage(frame_list)

    # PersonnelModel.train_and_save()
    # # VehicleModel.train_and_save()

    # print("Using ML Model to determine if a person is a military personnel or not from a video ...")
    # PersonnelModel.evaluate_frames(people_list)
    # # VehicleModel.evaluate_frames(vehicle_list)
    print("Using ML Model to determine if a person is a military personnel or not from an image ...")
    PersonnelModel.evaluate_image(people_list)
    
if __name__ == "__main__":
    main()
else:
    print("Error when importing")