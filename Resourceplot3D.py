
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_tracker_statistics_file(filename):    
    fp = open(filename,'r')
    times=[]
    areas=[]
    num_objs=[]
    fps=[]

    line_number = 0
    for line in fp.readlines():
        line_number += 1
        if 'ROI' in line:
            continue
        
        
        values = line.split(',')
        times.append(float(values[0]))
        areas.append(float(values[1]))
        num_objs.append(float(values[2]))
        fps.append(float(values[3]))

    return np.array(times),np.array(areas),np.array(num_objs),np.array(fps)

# times, areas, num_objs, fps = read_tracker_statistics_file('Statistics/tracker_statistics1.txt')
# mask = num_objs==16
# tmp_areas = areas[mask]
# idx = np.argsort(tmp_areas)
# tmp_areas = tmp_areas[idx]
# tmp_times = times[mask]
# tmp_times=tmp_times[idx]
# sorted_num_objs = num_objs[idx]
# sorted_times = times[idx]
# sorted_areas = areas[idx]

# fig = plt.figure()
# ax = fig.add_subplot(111)#
# ax.plot(tmp_areas, tmp_times)
# ax.set_xlabel("areas")
# ax.set_ylabel("execution time per frame (sec)")
# plt.title("Tracker Test in video 1- number of objects 16 ")
# plt.show()



def read_file(filename):
    fp = open(filename,'r')
    values = []
    line_number = -1
    for line in fp.readlines():
        line_number += 1
        if line_number ==0 or line_number == 1:
            continue

        values.append([float(i) for i in line.split(',')])

    return values
    
# start working with getting data    
v= []
filename = 'plot_info_2.txt'
v.append(read_file(filename))

# filename = 'C:\\Users\\Morteza\\Downloads\\plot_info_2.txt'
# v.append(read_file(filename))
# filename = 'C:\\Users\\Morteza\\Downloads\\plot_info_3.txt'
# v.append(read_file(filename))

fig = plt.figure()
ax = fig.gca(projection='3d')

Statistics = np.zeros((255,4)) # number_objsx[mem, cpu, time, F1]
x_axis = np.arange(255)
num_objs = []
F1s = []
times=[]
mem_means=[]
cpu_means=[]
mem_sums=[]
cpu_sums=[]
avg_areas=[]

for data in v:
    for d in data:
        F1s.append(d[0])
        times.append(d[1])
        mem_sums.append(d[2])
        mem_means.append(d[3])
        cpu_sums.append(d[4])
        cpu_means.append(d[5])
        num_objs.append(d[6])
        avg_areas.append(d[7])
        # idx = int(d[6])
        # Statistics[idx][0]=d[0]# F1
        # Statistics[idx][1]=d[1] # Time
        # Statistics[idx][2]=d[3] # Mem-mean
        # Statistics[idx][3]=d[5] # Cpu-mean
        
        
# plt.plot(x_axis, Statistics[:,0])
idx = np.argsort(num_objs)
num_objs = np.array(num_objs)
mem_means = np.array(mem_means)
mem_sums = np.array(mem_sums)
cpu_sums = np.array(cpu_sums)
cpu_means = np.array(cpu_means)
avg_areas = np.array(avg_areas)
times = np.array(times)
sorted_num_objs = num_objs[idx]
sorted_avg_areas = avg_areas[idx]
sorted_avg_areas = avg_areas[idx]
sorted_times = times[idx]
sorted_mem_means = mem_means[idx]
sorted_mem_sum = mem_sums[idx]
sorted_cpu_means = cpu_means[idx]
sorted_cpu_sums = cpu_sums[idx]
mask = num_objs==165#297#
areas_masked = avg_areas[mask]
idx = np.argsort(areas_masked)
# time_masked = time_masked[idx]
time_masked = times[mask]
cpu_sums_masked = cpu_sums[mask]
mem_sum_masked = mem_sums[mask]
mem_mean_masked = mem_means[mask]
cpu_mean_masked = cpu_means[mask]
fig = plt.figure()
ax = fig.add_subplot(111)#
# ax = fig.gca(projection='3d')
ax.plot(areas_masked[idx], cpu_mean_masked[idx])
ax.set_xlabel("average area in one segment")#accumulative sum of the number of objects in one segment")
ax.set_ylabel("accumulative sum of CPU usage in processing the frames of one segmen (ratio of memory usage to total memory)")
# ax.set_zlabel("Sum of cpu usage in processing one segment (percentage)")#accumulative sum of memory usage in processing the frames of one segment (sum of MBs)")#"mean of memory usage segment (MB) in one segment")
# ax.set_zlabel("consumed time")
ax.legend()
plt.title("pipeline test on video 1- area versus CPU consumption - number of objects in the segment 297 (8 segments)")
plt.show()
