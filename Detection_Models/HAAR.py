from setting import *
import dlib
import cv2

class C_DETECTION_HAAR:
    def __init__(self,file_address):
        self.__detector = cv2.CascadeClassifier(file_address)

    
    '''
        in_frame: input frame that can be grayscale or color image
        out_frame: output frame that can be a color output image when the input image is gray. So input and output image are considered as two different parameters
    '''
    def Detection_plus_BoundingBox(self, inp_frame):
        # Detects cars of different sizes in the input image
        cars = self.__detector.detectMultiScale(inp_frame, VEHICLE_HAAR_SCALE_FACTOR, VEHICLE_HAAR_MIN_NEIGHBORS)
        
        # To draw a rectangle in each cars
        for (x,y,w,h) in cars:
            cv2.rectangle(inp_frame,(x,y),(x+w,y+h),(0,0,255),2)
                
        return cars, inp_frame