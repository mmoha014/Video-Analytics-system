import numpy as np
import cv2

def correct_position(box):    
    x = int(round(abs(box[0])))
    y = int(round(abs(box[1])))
    w = int(round(abs(box[2])))
    h = int(round(abs(box[3])))
    startX, startY, endX, endY = x,y, x+w, y+h
    return [startX, startY, endX, endY]

def box_iou2(a, b):
        '''
        Helper funciton to calculate the ratio between intersection and the union of
        two boxes a and b
        a[0], a[1], a[2], a[3] <-> left, up, right, bottom
        '''                
        w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
        h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
        s_intsec = w_intsec * h_intsec
        s_a = (a[2] - a[0])*(a[3] - a[1])
        s_b = (b[2] - b[0])*(b[3] - b[1])
  
        return float(s_intsec)/(abs(s_a) + abs(s_b) -s_intsec)

        

def KalmanFilter_draw_box( img, trackers, colors):
        show_label = True
        for idx,trk in enumerate(trackers):
                '''
                Helper funciton for drawing the bounding boxes and the labels
                bbox_cv2 = [left, top, right, bottom]
                '''
                bbox_cv2 = trk.box
                label = trk.id                
                box_color = colors[idx]
                #box_color= (0, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_size = 0.7
                font_color = (0, 0, 0)
                # left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]
                left, top, right, bottom = bbox_cv2[0], bbox_cv2[1], bbox_cv2[2], bbox_cv2[3]
                
                # Draw the bounding box
                cv2.rectangle(img, (left, top), (right, bottom), box_color[0:3], 1)
                # cv2.putText(img, str(label), (left-2, top-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)
                
                if show_label:
                        # Draw a filled box on top of the bounding box (as the background for the labels)
                        cv2.rectangle(img, (left-2, top-20), (right+2, top), box_color[0:3], -1, 1)
                        
                        # Output the labels that show the x and y coordinates of the bounding box center.
                        # text_x= 'x='+str((left+right)/2)
                        # cv2.putText(img,text_x,(left,top-25), font, font_size, font_color, 1, cv2.LINE_AA)
                        # text_y= 'y='+str((top+bottom)/2)
                        cv2.putText(img,str(trk.id),(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)
                
        return img 