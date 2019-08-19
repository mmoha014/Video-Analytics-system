import cv2
# import dlib
# import numpy as np
from imutils import paths
from setting import *

        

class C_VIDEO_UNIT:
    '''
        input_video: file address or camera number
        output_video: where to save the output after processing the input video
        when video_write is True, the output_video is used as the output address for saving the frames after processing
    '''

    def __init__(self,input_video, video_write=False, output_video=None):
        self.__sequence_images_counter = -1        
        if type(input_video) is int:
            self.__capt = cv2.VideoCapture(input_video)
        elif(len(input_video.split("."))>1):
            #video address
            self.__capt = cv2.VideoCapture(input_video)
        else:
            #directory address
            self.__sequence_images_counter = 0
            self.__capt = sorted(paths.list_images(input_video))
        
        self.__VideoWriter = None
        if video_write == True:
            if self.__sequence_images_counter == -1:
                self.__frame_width = int(self.__capt.get(3))
                self.__frame_height = int(self.__capt.get(4))
            else:
                img = cv2.imread(self.__capt[0])
                self.__frame_width = int(img.shape[1])
                self.__frame_height = int(img.shape[0])
            self.__VideoWriter = cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc('M','J','P','G'), 15, (self.__frame_width, self.__frame_height))
            

    def get_frame(self):
        try:
            if self.__sequence_images_counter > -1:
                frame = cv2.imread(self.__capt[self.__sequence_images_counter])
                self.__sequence_images_counter += 1
                return True, frame
            else:
                success, frame = self.__capt.read()
                return success, frame
        except:
            print('Error in reading frame from source')
            return False, None

    
    def write_output_video(self, in_frame):

        if len(in_frame.shape)>2:
            # color image
            self.__VideoWriter.write(in_frame)
        else:
            # grayscale image
            self.__VideoWriter.write(cv2.cvtColor(in_frame,cv2.COLOR_GRAY2BGR))
            
    def __del__(self):
        if self.__sequence_images_counter==-1:
            self.__capt.release()
        if self.__VideoWriter != None:
            self.__VideoWriter.release()  

