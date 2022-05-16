import cv2
import imutils
import numpy as np
import argparse

class HOG_HumanDetection:
    HOGCV = None
    camera = None
    video = None
        
    def __init__(self, camera=False):
        self.HOGCV = cv2.HOGDescriptor()
        self.HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        self.camera = camera
        
    def __init__(self, video=False):
        self.HOGCV = cv2.HOGDescriptor()
        self.HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        self.video = video
        
        

    def detect(self, frame):
        bounding_box_cordinates, weights =  self.HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
        
        person = 1
        for x,y,w,h in bounding_box_cordinates:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            person += 1
        
        cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
        cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
        cv2.imshow('output', frame)
    
        return frame
    
    def detectByPathVideo(self, path, writer):
    
        video = cv2.VideoCapture(path)
        check, frame = video.read()
        if check == False:
            print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
            return
    
        print('Detecting people...')
        while video.isOpened():
            #check is True if reading was successful 
            check, frame =  video.read()
    
            if check:
                frame = imutils.resize(frame , width=min(800,frame.shape[1]))
                frame = self.etect(frame)
                
                if writer is not None:
                    writer.write(frame)
                
                key = cv2.waitKey(1)
                if key== ord('q'):
                    break
            else:
                break
        video.release()
        cv2.destroyAllWindows()
    
    
    def detectByPathImage(self, path, output_path):
        image = cv2.imread(path)
    
        image = imutils.resize(image, width = min(800, image.shape[1])) 
    
        result_image = self.detect(image)
    
        if output_path is not None:
            cv2.imwrite(output_path, result_image)
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    def humanDetector(self, args):
        image_path = args["image"]
        video_path = args['video']
    
        writer = None
        if args['output'] is not None and image_path is None:
            writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))
    
        if video_path is not None:
            print('[INFO] Opening Video from path.')
            self.detectByPathVideo(video_path, writer)
        elif image_path is not None:
            print('[INFO] Opening Image from path.')
            self.detectByPathImage(image_path, args['output'])
    
    def argsParser(self):
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
        arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
        arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
        args = vars(arg_parse.parse_args())
    
        return args
    
if __name__ == "__main__":
    humanDetector = HOG_HumanDetection()
    humanDetector.humanDetector(humanDetector.argsParser())
else:
    print("Execution fail when importing")