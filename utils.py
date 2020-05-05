import numpy as np
import cv2
import sys

#=================================== added in Feb 17, 2020 =================================
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def resize_image(img, target_size):
    return cv2.resize(img, (target_size, target_size))
# ====start================= version 3 ========================
def k_min_cell_index_3D_matrix(m,k):    
    min_values = np.ones(k, dtype=np.float)*100.  # check this value for the future
    min_values_arg = [[1,1,1] for i in range(k)]
    worst = []   
    max_value = 0
    for i in range(len(m)):
        for j in range(len(m[i])):
            for k in range(len(m[i][j])):
                
                if np.sum(m[i][j][k]<min_values)>0:
                    index = np.argmax(min_values)
                    min_values[index] = m[i][j][k]
                    min_values_arg[index] = [i,j,k]
                    #min_value = m[i][j][k] 
                    #qu.enqueue(min_value)  
                if m[i][j][k]>max_value:
                    worst = [i,j,k]
                    max_value = m[i][j][k]
                    
    index_sort = np.argsort(min_values)
    result = [min_values_arg[index] for index in index_sort]
    
    #return qu.get()  
    return result, np.sort(min_values), worst    
#====end ============ version 3 ===============================
                
def correct_position(box):    
    x = int(round(abs(box[0])))
    y = int(round(abs(box[1])))
    w = int(round(abs(box[2])))
    h = int(round(abs(box[3])))
    startX, startY, endX, endY = x,y, x+w, y+h
    return [startX, startY, endX, endY]

def box_iou2(a, b):
    # print("a:",str(a))
    # print("b:",str(b))
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''                
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    # print("w_insect: ", w_intsec, ", h_insect: ", h_intsec)
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])
    # print("s_a:",str(s_a),", s_b: ",str(s_b))
    # print(str(s_intsec)+"/"+str(abs(s_a))+" + "+str(abs(s_b)+" - "+str(s_intsec)))
    # print("s_intesc:",s_intsec," s_a: ", s_a, "s_b: ",s_b)
    # print(str(float(s_intsec))+"/"+str((abs(s_a) + abs(s_b) -s_intsec)))
    # if s_intsec == 0:
    return float(s_intsec)/(abs(s_a) + abs(s_b) -s_intsec) if s_intsec!=0 else 0

        
def resize_bounding_box(bboxes, scale):
    return np.array(bboxes) /scale

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

def draw_MOT_box(boxes, frame):
    for box in boxes:
        x = int(round(box[0]))
        y = int(round(box[1]))
        w = int(round(box[2]))
        h = int(round(box[3]))
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,20,111), 2)
    
    return frame

def draw_box(boxes, frame):
    for box in boxes:
        x = int(round(box[0]))
        y = int(round(box[1]))
        w = int(round(box[2]))
        h = int(round(box[3]))
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,20,111), 2)
    
    return frame

"""
input_video: input video address
output_Dir: Don't use '/' at the end of the  output directory address where the frames will be stored. 
"""
def Video_2_image(input_video, output_Dir, IsGrayscale=False):
    capt = cv2.VideoCapture(input_video)
    frame_number = 0
    while True:
        ret, frame = capt.read()
        if not ret:
            break

        if IsGrayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(output_Dir+"/"+str(frame_number)+".jpg", frame)
        frame_number += 1

