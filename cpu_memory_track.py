import time
import multiprocessing as mp
import psutil
import numpy as np
from pipeline2 import main
from GUI_Pipeline import main_APP, App
import tkinter
def monitor(target):
    worker_process = mp.Process(target = target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)

    #log cpu usage of worker process every 10 ms
    cpu_percents = []
    mem_usage = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent()/psutil.cpu_count())
        mem_usage.append(p.memory_info().rss)
        time.sleep(0.01)
    
    worker_process.join()
    return cpu_percents, mem_usage

cpu_percent, mem_use= monitor(App(tkinter.Tk(), "Dynamic change of modules in Tracking"))#main)#
print('cpu usage:',np.average(cpu_percent))
print(('memory usage',np.average(mem_use)/1024)/1024)