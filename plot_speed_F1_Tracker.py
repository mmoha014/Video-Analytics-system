import numpy as np
import matplotlib.pyplot as plt
#===================================== ExecutionTime vs Tracker ===================================
fp = open('features_v2_NEW.txt')
# /home/mgharasu/Documents/Purified codes/features_v1_NEW.txt
# /home/mgharasu/Documents/Purified codes/features_v2_NEW.txt
tracker = None
isObjD = None
image_size = None
frame_rate = None
is_config_read = True
is_metrics_read = False
is_read_conflict = False
is_read_area = False
bbox = []
loop = 0
VELOCITY = {'CSRT':[],'MEDIANFLOW':[], 'MOSSE':[],'KCF':[],'objD':[]}
F1 = {'CSRT':[],'MEDIANFLOW':[], 'MOSSE':[],'KCF':[],'objD':[]}
data_execTime = []
data_tracker = []
data_num_obj = []
frame_rate_fix = 30
frame_size_fix = 960
for l in fp.readlines():
    if 'Deep_Yolo' in l:
        values = l.split(',')
        tracker_type = values[-1][2:-3]
        frame_size = int(values[0][1:])
        frame_rate = int(values[1])
        is_config_read = False
        is_metrics_read = True
        continue
    
    if 'objD' in l or 'tr' in l:
        metrics = l.split(',')
        num_objs = int(metrics[0])
        execTime = float(metrics[1])
        f1 = float(metrics[2])
        cpu = float(metrics[3])
        mem_parent = int(metrics[4])
        mem_child = int(metrics[5])
        isObjD = metrics[6][1:-1]
        is_bbox_read = True        
        is_metrics_read = False
        continue

    if is_bbox_read:   
        if num_objs>loop and num_objs>0:
            v = l.split(',')
            bbox.append([float(v[0][1:]),float(v[1]),float(v[2]),float(v[3]),float(v[4][:-2])])
            loop += 1
            continue
        else:            
            if tracker_type=='CSRT' and f1<.4 and frame_rate==30 and frame_size==960:
                print(bbox)
                
        bbox = []
        is_bbox_read = False
        is_read_conflict = True
        loop = 0
    
    # reading conflict
    if is_read_conflict:
        conflict =[]
        v=l.split(',')
        if '[]' in v[0]:
            conflict=[0]
        else:
            v[0]=v[0][1:]
            v[-1]=v[-1][:-2]
            for va in v:
                conflict.append(float(va))#[float(va[0]),float(va[1]),float(va[2]),float(va[3]),float(va[4])])
        is_read_conflict = False
        is_read_velocity = True
        continue
    
    if is_read_velocity:        
        velocity = []
        values = l.split(',')
        if '[]' in values[0]:
            velocity.append(0)
        else:
            values[0] = values[0][1:]
            values[-1] = values[-1][:-2]
            for v in values:
                velocity.append(float(v))
        
        if frame_rate==30 and frame_size==960 and f1>0.1:
            VELOCITY[tracker_type].append(np.max(velocity))
            F1[tracker_type].append(f1)
        is_read_velocity = False
        is_read_area = True

        velocity = []
        continue

    if is_read_area:
        area = []
        values = l.split(',')
        if '[]' in values:
            velocity.append(0)
        else:
            values[0] = values[0][1:]
            values[-1] = values[-1][:-2]
            for v in values:
                area.append(float(v))
        # is_read_velocity = False
        is_read_area = False
        is_metrics_read = True
        continue

dvsn = 10
data_CSRT = [None for i in range(int(max(VELOCITY['CSRT']))//dvsn)]
CSRT = np.array(VELOCITY['CSRT'])
for i in range(len(data_CSRT)):
    indices = np.where(CSRT>=i*dvsn) and np.where(CSRT<(i+1)*dvsn)
    data_CSRT[i]= np.array(F1['CSRT'])[indices]

data_MEDIANFLOW = [None for i in range(int(len(VELOCITY['MEDIANFLOW']))//dvsn)]
for i in range(len(data_MEDIANFLOW)):
    indices = np.where(np.array(VELOCITY['MEDIANFLOW'])>=i) and np.where(np.array(VELOCITY['MEDIANFLOW'])<(i+1)*dvsn)
    data_MEDIANFLOW[i]= np.array(F1['MEDIANFLOW'])[indices]

data_MOSSE = [None for i in range(int(len(VELOCITY['MOSSE']))//dvsn)]
for i in range(len(data_MOSSE)):
    indices = np.where(np.array(VELOCITY['MOSSE'])>=i*dvsn) and np.where(np.array(VELOCITY['MOSSE'])<(i+1)*dvsn)
    data_MOSSE[i]= np.array(F1['MOSSE'])[indices]

data_KCF = [None for i in range(int(len(VELOCITY['KCF']))//dvsn)]
for i in range(len(data_KCF)):
    indices = np.where(np.array(VELOCITY['KCF'])>=i*dvsn) and np.where(np.array(VELOCITY['KCF'])<(i+1)*dvsn)
    data_KCF[i]= np.array(F1['KCF'])[indices]

data_objD = [None for i in range(int(len(VELOCITY['objD']))//dvsn)]
for i in range(len(data_objD)):
    indices = np.where(np.array(VELOCITY['objD'])>=i*dvsn) and np.where(np.array(VELOCITY['objD'])<(i+1)*dvsn)
    data_objD[i]= np.array(F1['objD'])[indices]


def draw_plot2(data, offset,edge_color, fill_color):
    pos = np.arange(data.shape[0])+offset 
    bp = ax.boxplot(data, positions= pos, widths=0.3, patch_artist=True, manage_xticks=False)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


fig, ax = plt.subplots()
draw_plot2(np.array(data_CSRT), -0.9, "red", "white")

draw_plot2(np.array(data_KCF), -0.5, "green", "white")
draw_plot2(np.array(data_MEDIANFLOW), 0.0, "blue", "white")
draw_plot2(np.array(data_MOSSE), 0.5, "violet", "white")
# draw_plot2(np.array(data_objD), 0.9, "black", "white")
plt.show()