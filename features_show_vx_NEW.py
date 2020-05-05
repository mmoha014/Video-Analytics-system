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
OBJS = {'CSRT':[],'MEDIANFLOW':[], 'MOSSE':[],'KCF':[],'objD':[]}
ExTime = {'CSRT':[],'MEDIANFLOW':[], 'MOSSE':[],'KCF':[],'objD':[]}
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
        F1 = float(metrics[2])
        cpu = float(metrics[3])
        mem_parent = int(metrics[4])
        mem_child = int(metrics[5])
        isObjD = metrics[6][1:-1]
        # is_bbox_read = True        
        # is_metrics_read = False
        if frame_rate == frame_rate_fix and frame_size_fix==frame_size:
            # data_num_obj.append(num_objs)
            if isObjD == 'tr':
                OBJS[tracker_type].append(num_objs)
                ExTime[tracker_type].append(execTime)
            else:
                OBJS[isObjD].append(num_objs)
                ExTime[isObjD].append(execTime)
            # data_tracker.append(isObjD)
            # data_execTime.append(execTime)
        continue
data_CSRT = [None for i in range(max(OBJS['CSRT'])+1)]
for i in range(max(OBJS['CSRT'])+1):
    indices = np.argwhere(np.array(OBJS['CSRT'])==i)
    data_CSRT[i]= np.array(ExTime['CSRT'])[indices]

data_objD = [None for i in range(max(OBJS['objD'])+1)]
for i in range(max(OBJS['objD'])+1):
    indices = np.argwhere(np.array(OBJS['objD'])==i)
    data_objD[i]= np.array(ExTime['objD'])[indices]

data_MEDIANFLOW = [None for i in range(max(OBJS['MEDIANFLOW'])+1)]
for i in range(max(OBJS['MEDIANFLOW'])+1):
    indices = np.argwhere(np.array(OBJS['MEDIANFLOW'])==i)
    data_MEDIANFLOW[i]= np.array(ExTime['MEDIANFLOW'])[indices]

data_MOSSE = [None for i in range(max(OBJS['MOSSE'])+1)]
for i in range(max(OBJS['MOSSE'])+1):
    indices = np.argwhere(np.array(OBJS['MOSSE'])==i)
    data_MOSSE[i]= np.array(ExTime['MOSSE'])[indices]

data_KCF = [None for i in range(max(OBJS['MOSSE'])+1)]
for i in range(max(OBJS['KCF'])+1):
    indices = np.argwhere(np.array(OBJS['KCF'])==i)
    data_KCF[i]= np.array(ExTime['KCF'])[indices]

endpint=0    
def draw_plot1(data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    plt.xticks(range(11))
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

def draw_plot2(data, offset,edge_color, fill_color):
    pos = np.arange(data.shape[0])+offset 
    bp = ax.boxplot(data, positions= pos, widths=0.3, patch_artist=True, manage_xticks=False)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

fig, ax = plt.subplots()
draw_plot2(np.array(data_CSRT), -0.4, "red", "white")
draw_plot2(np.array(data_objD), -0.2, "blue", "white")
draw_plot2(np.array(data_MEDIANFLOW), 0, "green", "white")
draw_plot2(np.array(data_MOSSE), 0.2, "violet", "white")
draw_plot2(np.array(data_KCF), 0.4, "black", "white")
# ax.legend(ncol=5, bbox_to_anchor=(0,1),loc='upper right', fontsize='small')
plt.show()
"""    
    if is_bbox_read:
        
        if num_objs>=loop:
            v = l.split(',')
            bbox.append([float(v[0][1:]),float(v[1]),float(v[2]),float(v[3]),float(v[4][:-2])])
            loop += 1
            continue
        else:
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
        is_read_velocity = False
        is_read_area = True
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
        


    """