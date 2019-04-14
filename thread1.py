# -*- coding: utf-8 -*-
import random
import time
from threading import Thread
from threading import Lock
data_upd1 = 0
data_upd2 = 0
lock1 = Lock() #общение с мастером
lock2 = Lock()
data1 = ''
data2 = ''
testcounter=0
conv = 0

def reading_dataset(file_name, n, d):
    out = [0] * n
    for i in range(n):
        out[i] = [0] * (d+1)
    f = open(file_name, 'r')
    for i in range(n):
        s=f.readline()
        tokens=s.split(' ')
        out[i][0]=tokens[0]
        del tokens[0]
        while len(tokens)>1:
            num_and_val=tokens[0].split(':')
            out[i][ int(num_and_val[0]) ]= num_and_val[1]
            del tokens[0]
    return out

def is_conv():
    global testcounter
    testcounter = testcounter + 1
    if testcounter==100:
        return 1
    return 0
class Slave(Thread):

    def __init__(self, name, xm):
        Thread.__init__(self)
        self.name = name
        self.xm = xm
        self.x = xm

    def run(self):
        global data_upd1
        global data_upd2
        global lock1
        global lock2
        global data1
        global data2
        global conv
        while 1:
            check = 0
            while 1:
                lock1.acquire()
                if data_upd1 == 0:
                    data_upd1 = 1
                    data1 = self.name
                    check = 1
                lock1.release()
                if check ==1 or conv==1:
                    break
            while 1:
                lock2.acquire()
                if data_upd2==1 and self.name==data2:
                    print(data2,self.name)
                    data_upd2=0
                    check=0
                lock2.release()
                if check==0 or conv==1:
                    break
            if conv==1:
                break


def master():
    my_threads = []
    global data_upd1
    global data_upd2
    global lock1
    global lock2
    global data2
    global data1
    global conv
    tmp=''
    out = reading_dataset("/home/lemikhovalex/Documents/6th term/opt/covtype.libsvm.binary.scale", 581012, 54)
    x1 = x2 = out[0]
    for i in range(100):
        name = "Slave #%s" % (i + 1)
        my_threads.append(Slave(name,x0))
        my_threads[i].start()
    while 1:
        check = 0
        while 1:
            lock1.acquire()
            if data_upd1 == 1:
                tmp=data1
                data_upd1 = 0
                check = 1
            lock1.release()
            if check ==1:
                break
        while 1:
            lock2.acquire()
            if data_upd2==0:
                data2 = tmp
                data_upd2=1
                check=0
            lock2.release()
            if check==0:
                break
        if is_conv()==1:
            conv = 1
            break
    for i in range(100):
        my_threads[i].join()


if __name__ == "__main__":
    master()