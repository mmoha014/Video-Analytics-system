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

class App:
    def __init__(self, window, window_title, video_source=0):                                     
        # open video source (by default this will try to open the computer webcam)
        self.video = C_VIDEO_UNIT(INPUT_VIDEO_SNOURCE)#"/home/morteza/Desktop/Dr. Jain/project implementation/week1/outputs June 27/Haar_video_out.avi")
        self.__detection = C_DETECTION(DETECTION_METHOD)
        self.MOT = C_MOT_OUTPUT_GENERATER(FOLDER+'.txt')
        self.tracker = []
        self.frame_number = -1
        self.frame_size = []
        self.frame_size = None
        self.search_for_frame_to_detect_object = True
        self.frame = None
        self.old_detboxes = []
        self.istracking = True
        self.isTracker = False
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
        helpmenu.add_command(label="No Tracker", command=self.No_Tracker)
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
        self.isTracker = False
        self.__detection.Switch_detection("Deep_Yolo")
    
    def Speed(self):
        self.isTracker = False
        self.__detection.Switch_detection("HOG_Pedestrian")
    
    def KCF(self):
        self.isTracker = False
        self.tracker.switch_Tracker(self.frame, "KCF")
    
    def MOSSE(self):
        self.isTracker = False
        self.tracker.switch_Tracker(self.frame, "MOSSE")

    def MEDIANFLOW(self):
        self.isTracker = False
        self.tracker.switch_Tracker(self.frame, "MEDIANFLOW")
    
    def CSRT(self):
        self.isTracker = False
        self.tracker.switch_Tracker(self.frame, "CSRT")
    
    def No_Tracker(self):
        self.isTracker = False
        print("No Tracker")
    



    # considering the pipeline here
    def update(self):
        
        # Get a frame from the video source        
        ret, self.frame = self.video.get_frame()        
        if not ret:
            print("End of Program")
            quit()
        
        self.frame = cv2.resize(self.frame, (int(self.frame.shape[1]/2), int(self.frame.shape[0]/2)))

        self.frame_number += 1
        
        if self.isTracker:
            if self.frame_number % UPDATE_TRACKER == 0:
                self.search_for_frame_to_detect_object = True            

            if self.search_for_frame_to_detect_object:
                detected_boxes, self.frame = self.__detection.Detection_BoundingBox(self.frame)

                if len(detected_boxes)>0:
                    self.search_for_frame_to_detect_object = False
                    #multi tracker
                    if self.tracker == []:
                        self.tracker = C_TRACKER(TRACKER_TYPE)
                        self.tracker.Add_Tracker(self.frame, detected_boxes)                    
                        #self.MOT.write(self.frame_nmber, self.tracker.Get_MOTChallenge_Format())

                        #continue
                        self.window.after(self.delay, self.update)
                        return
                    else:
                        self.tracker.update_pipeline(self.frame, detected_boxes)
                        # self.MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
                        
                        #continue
                        self.window.after(self.delay, self.update)
                        return

                # Updating tracker when object detectin looking for objects in frame if object detection method fails to detect. In this situation, tracker keeps on his task.
                # We must not leave a frame without tracking
                if self.tracker != []:
                    self.frame = self.tracker.update(self.frame)
                    # self.MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
                
                self.show_in_window(self.frame)
                #continue
                self.window.after(self.delay, self.update)
                return
            
            else:
                self.frame = self.tracker.update(self.frame)
                # self.MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
                self.show_in_window(self.frame)
                    
            
            self.show_in_window(self.frame)
        
        else: #isTracking==False
            predicted_box, self.frame = self.__detection.Detection_BoundingBox(self.frame)
            if self.old_detboxes is None:
                self.old_detboxes = predicted_box
            else:
                # do linear assignment
                matched, unmatched_new, unmatch_old = self.tracker.assign_detections_to_trackers(self.old_detboxes, predicted_box)

        self.window.after(self.delay, self.update)        

        
    def show_in_window(self, frame):
        # frame = cv2.resize(frame, (int(frame.shape[1]/2), frame.shape[0]/2))
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        
        # self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        if self.panel2 is None:
            self.panel2 = Label(image=self.photo)
            self.panel2.image = self.photo
            self.panel2.pack(side="bottom")                    
        else:
            self.panel2.configure(image=self.photo)
            self.panel2.image = self.photo                
            
 

if __name__ == "__main__":     
    # Create a window and pass it to the Application object
    App(tkinter.Tk(), "Tkinter and OpenCV")