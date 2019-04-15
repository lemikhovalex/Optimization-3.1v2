# -*- coding: utf-8 -*-
import random
import time
import scipy
import math
from threading import Thread
from threading import Lock

data_upd1 = 0
data_upd2 = 0
lock1 = Lock()  # общение с мастером
lock2 = Lock()
data1 = ''
data2 = ''
gldelta = 0.0
global glxm
testcounter = 0
conv = 0
epsilon_conv = 10 ** -4
L = 21930585,25
p = 7
lambda_1=1000
cores=7 #для вычислений, один на главный

def add_lists(x, y):
    out=[0]* (len(x))
    for i in range(len(x)):
        out[i]=x[i]+y[i]
    return out

def scal_mul(x, z):
    global d
    out=0
    for i in range(d):
        out+=x[i+1]*z[i+1]
    return out


def l_j(x, z_j):
    #l_j  where i depends on argument
    global n
    return  (1/n)*math.log(1 + math.exp(-z_j[0] * scal_mul(x, z_j)))

def nabla_z_l_j(x, z_j):
    global d
    out=[0]*(d+1)
    out[0]=0
    for i in range(1, d+1):
        out[i]=(1/n)/(1 + 1 + math.exp(-z_j[0] * scal_mul(x, z_j)))*z_j[i]*z_j[0]
    return out



def reading_dataset(file_name, n, d):
    out = [0] * n
    for i in range(n):
        out[i] = [0] * (d+1)
    f = open(file_name, 'r')
    for i in range(n):
        s=f.readline()
        tokens=s.split(' ')
        out[i][0]=tokens[0]
        out[i][0]=-3+2*out[i][0]
        del tokens[0]
        while len(tokens)>1:
            print(tokens)
            num_and_val=tokens[0].split(':')
            out[i][ int(num_and_val[0]) ]= num_and_val[1]
            del tokens[0]
    return out



def local_norm_2(x):
    out=0
    for i in range(1,len(x) ):
        out+=x[i]**2
    return math.sqrt(out)


def dist_2(x1, x2):
    global d
    temp=[0]*(d+1)
    for i in range(1, d+1):
        temp[i]=x1[i]-x2[i]
    return local_norm_2(temp)

def local_norm_1(x):
    out=0
    for i in range(1,len(x) ):
        out+=math.fabs(x[i])
    return out

def gamma():
    global L
    return 1/L

def is_conv(x_curr, x_prev):
    if dist_2(x_curr - x_prev) < epsilon_conv:
        return True
    else:
        return False


def r(x):
    global n
    out=0
    for i in range(1, len(x) ):
        out+=x[i]**2
    out*= 1/(2*n)
    for i in range(1, len(x)):
        out+=math.fabs(x[i])*lambda_1
    return out


def f_for_prox_gr(z, x):
    out=0
    out+= 1/(2*gamma())*local_norm_2(x)**2
    out += r(z)
    return out


def n_i(slave_num):
    global n
    global cores
    n_i=int(n/cores)
    if(slave_num==cores):
        return (n - n_i*(cores-1) )
    else:
        return n_i

def pi(slave_num):
    global n
    return n_i(slave_num)/n


def prox(x):
    res=scipy.optimize.minimize(
                            f_for_prox_gr,
                            x,
                            args=x,
                            method='BFGS',
                            jac=None,
                            hess=None,
                            hessp=None,
                            bounds=None,
                            constraints=(),
                            tol=None,
                            callback=None,
                            options=None
                            )
    return res.x


def sub_in_xplus(slave_num, x):
    #xplus = z - sub_in_xplus(slave_num, z)
    global A
    global d
    out=[0]*(d+1)
    z_j=[0]*(d+1)
    for j in range(n_i(slave_num)):
        for i in range(1, d+1):
            z_j[i]=A[j][i]
        out+=nabla_z_l_j(x, z_j)
    out*=gamma()
    out/=n_i(slave_num)
    return out


class Slave(Thread):
    global A
    def __init__(self, name,num):
        Thread.__init__(self)
        self.name = name
        self.xm = A[0]
        self.x = A[0]
        self.num=num

    def run(self):
        global data_upd1
        global data_upd2
        global lock1
        global lock2
        global data1
        global data2
        global conv
        global p
        global gldelta
        global glxm
        delta = 0.0
        for i in range(p):
            z = prox(self.xm+delta)
            xplus = z - sub_in_xplus(self.num, self.xm)
            delta += pi(self.num)* (xplus-self.x)
            self.x = xplus
        while 1:
            check = 0
            while 1:
                lock1.acquire()
                if data_upd1 == 0:
                    data1 = self.name
                    gldelta = delta
                    data_upd1 = 1

                    check = 1

                lock1.release()
                if check == 1 or conv == 1:
                    break
            while 1:
                lock2.acquire()
                if data_upd2 == 1 and self.name == data2:
                    print(data2, self.name)
                    self.xm = glxm
                    data_upd2 = 0
                    check = 0
                lock2.release()
                if check == 0 or conv == 1:
                    break
            if conv == 1:
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
    global A
    global glxm
    global gldelta
    x1 = x2 = A[0]
    delta = 0.0
    k = 0
    tmp = ''
    for i in range(8):
        name = "Slave #%s" % (i + 1)
        my_threads.append(Slave(name, i+1))
        my_threads[i].start()
    while 1:
        check = 0
        while 1:
            lock1.acquire()
            if data_upd1 == 1:
                tmp = data1
                delta = gldelta
                data_upd1 = 0
                check = 1
            lock1.release()
            if check == 1:
                break
        x2 = x2 + delta
        while 1:
            lock2.acquire()
            if data_upd2 == 0:
                data2 = tmp
                glxm = x2
                data_upd2 = 1
                check = 0
            lock2.release()
            if check == 0:
                break
        k = k + 1
        if k%8==0:
            if is_conv(x1,x2) == 1:
                conv = 1
                print(x1)
                break
            x1=x2
    for i in range(8):
        my_threads[i].join()


if __name__ == "__main__":
    A = reading_dataset("/home/lemikhovalex/Documents/6th term/opt/covtype.libsvm.binary.scale", 581012, 54)
    master()
