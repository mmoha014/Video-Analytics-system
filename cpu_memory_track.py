import time
import cv2
import multiprocessing as mp
import psutil
# from test import adds
import numpy as np
# from setting import manager
# from pipeline2 import main
# from GUI_Pipeline import main_APP, App
# import tkinter
from psutil import virtual_memory
from threading import Thread

# from profiling_Pipeline import 
# from datetime import datetime, timedelta
# from parser import parser
       
def monitor(target, args):
    # mp.set_start_method('spawn')
    total_memory = virtual_memory()[0]
    worker_process = mp.Process(target = target, args = args)#.start()
    # print("\n1\n")
    worker_process.start()
    # print("\n2\n")
    # print(worker_process)
    # worker_process.join()

    p = psutil.Process(worker_process.pid)
    # p.get_cpu_percent(interval=0.01)
    # print("\nthe function had run\n")
    #log cpu usage of worker process every 10 ms
    cpu_percents = []
    mem_usage = []
    while worker_process.is_alive():
        # print("\n4\n")    
        cpu_percents.append(p.cpu_percent()/psutil.cpu_count()/100)
        # print(p.cpu_percent)
        mem_usage.append(p.memory_info().rss/total_memory)
        # print('clock')
        time.sleep(0.01)
        

    worker_process.join()
    # worker_process.terminate()
    
    return cpu_percents, mem_usage

def monitor_wo_params(target):
    # mp.set_start_method('spawn')
    
    worker_process = mp.Process(target = target)#.start()
    # print("\n1\n")
    worker_process.start()
    # print("\n2\n")
    # print(worker_process)
    # worker_process.join()

    p = psutil.Process(worker_process.pid)
    # p.get_cpu_percent(interval=0.01)
    # print("\nthe function had run\n")
    #log cpu usage of worker process every 10 ms
    cpu_percents = []
    mem_usage = []
    while worker_process.is_alive():
        # print("\n4\n")
        cpu_percents.append(p.cpu_percent()/psutil.cpu_count())
        # print(p.cpu_percent)
        mem_usage.append(p.memory_info().rss)
        # print('clock')
        time.sleep(0.01)
        

    worker_process.join()
    # worker_process.terminate()
    
    return cpu_percents, mem_usage

#=========================================== additional code ========================================
"""

from parsers import parser
import psutil
from datetime import datetime, timedelta
class memtest(parser.parser):
    def __init__(self):
        parser.parser.__init__(self)
        self.lastMemUpdate = None
        self.totalMem = 0
        self.totalSwap = 0

    def do(self, data, mode):
        if self.pid != 0:
            if self.lastMemUpdate is None or datetime.now() - self.lastMemUpdate > timedelta(seconds=5):
                self.lastMemUpdate = datetime.now()
                p = psutil.Process(self.pid)

                parentMeminfo = p.memory_full_info()
                self.totalMem = parentMeminfo.rss
                self.totalSwap = parentMeminfo.swap
                for child in p.children(True):
                    childMemInfo = child.memory_full_info()
                    self.totalMem += childMemInfo.rss
                    self.totalSwap += childMemInfo.swap
        output = []
        for line in data.replace("\r", "\n").split("\n"):
            if len(line):
                output.append("[MEM:{:_>7.0f}MB,SWAP:{:_>7.0f}MB]{}".format(self.totalMem / (1024 * 1024), self.totalSwap / (1024 * 1024), line))
        return "\n".join(output)

[MEM:____349MB,SWAP:______0MB]Starting V-Ray Benchmark...
[MEM:____349MB,SWAP:______0MB]AMD Ryzen 5 1600 Six-Core Processor, # of logical cores: 12
[MEM:____349MB,SWAP:______0MB]NVIDIA driver version: 384.111
[MEM:____349MB,SWAP:______0MB]Ubuntu 17.10
[MEM:____349MB,SWAP:______0MB]V-Ray 3.57.01
[MEM:____349MB,SWAP:______0MB]Preparing to render on CPU...
[MEM:____349MB,SWAP:______0MB]Now rendering...
[MEM:___1075MB,SWAP:______0MB]Rendered 0%
[MEM:___1075MB,SWAP:______0MB]Rendered 0%
[MEM:___1075MB,SWAP:______0MB]Rendered 0%
[MEM:___1075MB,SWAP:______0MB]Rendered 0%
[MEM:___1075MB,SWAP:______0MB]Rendered 1%
[MEM:___1075MB,SWAP:______0MB]Rendered 1%
[MEM:___1075MB,SWAP:______0MB]Rendered 1%
[MEM:___1075MB,SWAP:______0MB]Rendered 1%
[MEM:___1075MB,SWAP:______0MB]Rendered 2%
[MEM:___1075MB,SWAP:______0MB]Rendered 2%
[MEM:___1075MB,SWAP:______0MB]Rendered 2%
[MEM:___1075MB,SWAP:______0MB]Rendered 2%
[MEM:___1075MB,SWAP:______0MB]Rendered 2%
[MEM:___1086MB,SWAP:______0MB]Rendered 3%
[MEM:___1086MB,SWAP:______0MB]Rendered 3%
[MEM:___1086MB,SWAP:______0MB]Rendered 3%
[MEM:___1086MB,SWAP:______0MB]Rendered 3%
[MEM:___1086MB,SWAP:______0MB]Rendered 4%
[MEM:___1086MB,SWAP:______0MB]Rendered 4%
[MEM:___1086MB,SWAP:______0MB]Rendered 4%
[MEM:___1086MB,SWAP:______0MB]Rendered 5%
[MEM:___1086MB,SWAP:______0MB]Rendered 5%
[MEM:___1086MB,SWAP:______0MB]Rendered 6%
[MEM:___1086MB,SWAP:______0MB]Rendered 6%
[MEM:___1086MB,SWAP:______0MB]Rendered 6%
[MEM:___1091MB,SWAP:______0MB]Rendered 6%
[MEM:___1091MB,SWAP:______0MB]Rendered 7%
[MEM:___1091MB,SWAP:______0MB]Rendered 7%
"""
# ====================================================================================================
# cpu_percent, mem_use= monitor(main)#App(tkinter.Tk(), "Dynamic change of modules in Tracking"))#
# print('cpu usage:',np.average(cpu_percent)/100)
# total_memory = np.divide(virtual_memory().total, np.power(1024,2), dtype=np.float) #version 3
# print('memory usage',np.divide(np.average(mem_use)/1024/1024,total_memory,dtype=np.float))
# ====================================================================================================

# """
# def measure(target, S,C,profiling):
#     cpu_percent, mem_use = monitor(target(S,C,profiling))
#     return np.average(cpu_percent), (np.average(mem_use)/1024/1024)


# def measure_adds(target, a,b):
#     cpu,mem = monitor(adds)
#     cpu_usage = np.average(cpu)
#     mem_usage = np.average(mem)/2048
# """


# def measure(target):
if __name__ == "__main__":
    a=0
    # monitor(adds(4,3))
    from lane_test import main
    # cpu, mem
    a=2
    cpu_percent, mem_usage = monitor_wo_params(main)
    # print np.average(cpu_percent), (np.average(mem_use)/1024/1024)
    print('cpu: ',np.average(cpu_percent), ', mem: ',(np.average(mem_usage)/1024/1024))
