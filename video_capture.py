import cv2
# import dlib
# import numpy as np
from imutils import paths
from setting import *
import time
        

class C_VIDEO_UNIT:
    '''
        input_video: file address or camera number
        output_video: where to save the output after processing the input video
        when video_write is True, the output_video is used as the output address for saving the frames after processing
    '''

    def __init__(self,input_video, video_write=False, output_video=None, frame_rate=30):
        self.__input_video = input_video
        self.__sequence_images_counter = -1
        video_type = 0
        self._number_of_frames = 0
        if type(input_video) is int:
            self.__capt = cv2.VideoCapture(input_video)
            video_type = 0
        elif(len(input_video.split("."))>1):
            #video address
            video_type = 1
            self.__capt = cv2.VideoCapture(input_video)
            self.__frame_rate = self.__capt.get(cv2.CAP_PROP_FPS)
        else:
            #directory address
            video_type = 2
            self.__sequence_images_counter = 0
            self.__capt = sorted(paths.list_images(input_video))
            self._number_of_frames = len(self.__capt)
        
        self.__frame_rate = frame_rate#self.get_frame_rate(input_video, video_type)
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
            
    
    def get_number_of_frames(self):
        return self._number_of_frames
        
    def get_frame_position(self, start_frame):
        try:
            frame = cv2.imread(self.__capt[start_frame])
            return True, frame
        except:
            return False, frame
        
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
            

    def get_frame_rate(self, input_video, video_type):
        capt = cv2.VideoCapture(input_video)
        if(video_type == 1):
            # video_file
            return capt.get(cv2.CAP_PROP_FPS)
        else:
            # Find OpenCV version
            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
            
            # With webcam get(CV_CAP_PROP_FPS) does not work.
            # Let's see for ourselves.
            
            if int(major_ver)  < 3 :
                fps = capt.get(cv2.cv.CV_CAP_PROP_FPS)
                print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
            else :
                fps = capt.get(cv2.CAP_PROP_FPS)
                print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
            
            # Number of frames to capture
            num_frames = 120;
            # Start time
            start = time.time()
            
            # Grab a few frames
            for i in range(0, num_frames) :
                ret, frame = capt.read()
        
            
            # End time
            end = time.time()
        
            # Time elapsed
            seconds = end - start            
        
            # Calculate frames per second
            fps  = num_frames / seconds;

            return fps


    def __del__(self):
        if self.__sequence_images_counter==-1:
            self.__capt.release()
        if self.__VideoWriter != None:
            self.__VideoWriter.release()  

