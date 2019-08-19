from tkinter import *
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from video_capture import C_VIDEO_UNIT
from tracking import C_TRACKER
from Object_detection import C_DETECTION
from MOT_File_Generator import C_MOT_OUTPUT_GENERATER # version 2
from setting import *
from utils import KalmanFilter_draw_box
import time
import numpy as np

class App:
    def __init__(self, window, window_title, video_source=0):                                     
        # open video source (by default this will try to open the computer webcam)
        #=========== begin of time related attributes ============
        t1 = time.time()
        self.video = C_VIDEO_UNIT(INPUT_VIDEO_SNOURCE)#"/home/morteza/Desktop/Dr. Jain/project implementation/week1/outputs June 27/Haar_video_out.avi")
        t2 = time.time()
        self.time_init_get_frame = t2-t1

        t1 = time.time()
        self.__detection = C_DETECTION(DETECTION_METHOD)
        t2 = time.time()
        self.time_init_detection = t2-t1

        self.time_avg_tracker = []
        self.time_avg_detection = []
        self.time_init_tracker = 0
        self.time_avg_get_frame = []
        # show results related to time
        self.font = cv2.FONT_ITALIC
        self.font_size = 0.4
        self.font_color = (255, 255, 255)
        #=========== end of time related attributes ============

        self.MOT = C_MOT_OUTPUT_GENERATER(FOLDER+'.txt')
        self.tracker = None
        self.frame_number = -1
        self.frame_size = []
        self.frame_size = None
        self.search_for_frame_to_detect_object = True
        self.frame = None
        self.tracker_type = TRACKER_TYPE
        
        # Create a canvas that can `fit the above video source size
        self.window = window        
        self.window.title(window_title)
        self.panel2 = None        

        menubar = Menu(window)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Accuracy", command=self.Accuracy)
        filemenu.add_command(label="Speed", command=self.Speed)
        menubar.add_cascade(label="Detection", menu=filemenu)

        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="CSRT", command=self.CSRT)
        helpmenu.add_command(label="KCF", command=self.KCF)
        helpmenu.add_command(label="MOSSE", command=self.MOSSE)
        helpmenu.add_command(label="MEDIANFLOW", command=self.MEDIANFLOW)
        helpmenu.add_command(label="Kalman Filter Association", command=self.No_Tracker)
        menubar.add_cascade(label="Tracker", menu=helpmenu)
        
        exitmenu = Menu(menubar, tearoff=0)
        exitmenu.add_command(label="Quit", command=window.quit)        
        menubar.add_cascade(label="Quit", menu=exitmenu)
        window.config(menu=menubar)                                

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        
        self.update()
        # self.window.before()
        self.window.mainloop()
        self.MOT.close()
    



    def Accuracy(self):
        self.__detection.Switch_detection("Deep_Yolo")
    
    def Speed(self):        
        self.__detection.Switch_detection("HOG_Pedestrian")
    
    def KCF(self):
        self.tracker_type = "KCF"
        self.tracker.switch_Tracker(self.frame, self.tracker_type)
    
    def MOSSE(self):
        self.tracker_type = "MOSSE"        
        self.tracker.switch_Tracker(self.frame, self.tracker_type)

    def MEDIANFLOW(self):
        self.tracker_type = "MEDIANFLOW"
        self.tracker.switch_Tracker(self.frame, self.tracker_type)
    
    def CSRT(self):
        self.tracker_type = "CSRT"
        self.tracker.switch_Tracker(self.frame, self.tracker_type)
    
    def No_Tracker(self):
        self.tracker_type = "Kalman_Filter"
        self.tracker.switch_Tracker(self.frame, self.tracker_type)        
    



    # considering the pipeline here
    def update(self):
        
        # Get a frame from the video source
        t1 = time.time()       
        ret, self.frame = self.video.get_frame()
        t2 = time.time()
        self.time_avg_get_frame.append(t2-t1)

        if not ret:
            print("End of Program")
            return
        
        self.frame = cv2.resize(self.frame, (int(self.frame.shape[1]/2), self.frame.shape[0]/2))

        self.frame_number += 1


        if self.frame_number % UPDATE_TRACKER == 0 or self.tracker_type is "Kalman_Filter":
            self.search_for_frame_to_detect_object = True            

        if self.search_for_frame_to_detect_object:
            t1 = time.time()
            detected_boxes, self.frame = self.__detection.Detection_BoundingBox(self.frame)            
            t2 = time.time()
            self.time_avg_detection.append(t2-t1)

            if self.tracker_type is "Kalman_Filter":
                if self.tracker is None:
                    self.tracker = C_TRACKER(self.tracker_type)
                trackers, colors = self.tracker.update_pipeline(self.frame, detected_boxes)
                frame = KalmanFilter_draw_box(self.frame, trackers, colors)
                self.show_in_window(self.frame)
                self.window.after(self.delay, self.update)                
                return

            if len(detected_boxes)>0:
                self.search_for_frame_to_detect_object = False
                #multi tracker
                if self.tracker == None:
                    t1 = time.time()
                    self.tracker = C_TRACKER(TRACKER_TYPE)
                    self.tracker.Add_Tracker(self.frame, detected_boxes)
                    t2 = time.time()
                    self.time_init_tracker = t2-t1
                    #self.MOT.write(self.frame_nmber, self.tracker.Get_MOTChallenge_Format())

                    #continue
                    self.window.after(self.delay, self.update)
                    return
                else:
                    t1 = time.time()
                    self.tracker.update_pipeline(self.frame, detected_boxes)
                    t2 = time.time()
                    self.time_avg_tracker.append(t2-t1)
                    # self.MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
                    
                    #continue
                    self.window.after(self.delay, self.update)
                    return

            # Updating tracker when object detectin looking for objects in frame if object detection method fails to detect. In this situation, tracker keeps on his task.
            # We must not leave a frame without tracking
            if self.tracker != None:
                t1 = time.time()
                self.frame = self.tracker.update(self.frame)
                t2 = time.time()
                self.time_avg_tracker.append(t2-t1)
                # self.MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
            
            self.show_in_window(self.frame)
            #continue
            self.window.after(self.delay, self.update)
            return
        
        else:
            t1 = time.time()
            self.frame = self.tracker.update(self.frame)
            t2 = time.time()
            self.time_avg_tracker.append(t2-t1)
            # self.MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
            self.show_in_window(self.frame)
                
        
        self.show_in_window(self.frame)
        self.window.after(self.delay, self.update)        

        
    def show_in_window(self, frame):
        # frame = cv2.resize(frame, (int(frame.shape[1]/2), frame.shape[0]/2))
        cv2.rectangle(frame, (0,0), (360,50), (0,0,0), -1, 1)
        cv2.putText(frame,str(" initial_getvideo: "+str(round(self.time_init_get_frame,3))+ " ms ")+str(",  avg_getvideo: "+str(round(np.average(self.time_avg_get_frame),3))+" ms"),(5,12), self.font, self.font_size, self.font_color, 1, cv2.LINE_AA)
        cv2.putText(frame,str("initial_detection: "+str(round(self.time_avg_detection[0],3))+ " ms")+str(",  avg_detection: "+str(round(np.average(self.time_avg_detection),3))+" ms"),(5,25), self.font, self.font_size, self.font_color, 1, cv2.LINE_AA)
        cv2.putText(frame,str("  initial_tracker: "+str(round(self.time_init_tracker,3))+ " ms")+str(",    avg_tracker: "+str(round(np.average(self.time_avg_tracker),3))+" ms"),(5,38), self.font, self.font_size, self.font_color, 1, cv2.LINE_AA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        
        
        # self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        if self.panel2 is None:
            self.panel2 = Label(image=self.photo)
            self.panel2.image = self.photo
            self.panel2.pack(side="bottom")                    
        else:
            self.panel2.configure(image=self.photo)
            self.panel2.image = self.photo                
            
 
def main_APP():
     App(tkinter.Tk(), "Dynamic change of modules in Tracking")

if __name__ == "__main__":     
    # Create a window and pass it to the Application object
    main_APP()