import time
import numpy as np
import scipy.optimize as opt
import math
import threading
import time
start_time = time.time()
sleep_time = 0
n = 581
d = 500
n_to_read = 581012
d_to_read = 100
data_upd1 = 0
data_upd2 = 0
lock1 = threading.Lock()  # общение с мастером
lock2 = threading.Lock()
data1 = 0
data2 = 0
gldelta=np.zeros((d, 1))
glxm=np.zeros((d, 1))
testcounter = 0
conv = 0
epsilon_conv = 10**-1
L = 0
p = 1
cores = 2 #для вычислений, один на главный
x_init = np.zeros((d, 1))
for i in range(d):
    x_init[i][0] = 1

sub_opt_arr = []
time_arr = []






def scal_mul(x, z):
    global d
    out = 0
    for i in range(d):
        tmp = x[i]
        print(tmp)
        out += tmp*z[i]
    return out


def local_norm_2(x):
    global d
    out = 0
    for i in range(d):
        out += x[i][0]**2
    return math.sqrt(out)


def local_norm_1(x):
    global d
    out = 0
    for i in range(0, d):
        out += math.fabs(x[i][0])
    return out


def gamma():
    global L
    return 1.0/L


def is_conv(x_curr, x_prev):
    global x_star
    delta = local_norm_2(x_curr - x_star)
    if delta < epsilon_conv:
        return True
    else:
        return False





def st_string(slave_num):
    return slave_num*int(d/cores)

def fi_string(slave_num):
    if slave_num!=cores:
        return (slave_num + 1)*int(d/cores)-1
    return d-1



def Slave(name, num):
    global A
    global d
    global x_init
    xm = x_init
    x = x_init
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
    global sleep_time

    while 1:
        x0 = xm
        delta = np.zeros((d, 1))

        for i in range(p):
            tmp = np.zeros((d, 1))
            for e in range(st_string(num),fi_string(num)+1):
                for j in range(d):
                    tmp[e] += B[e][j]*xm[j]
                tmp[e]-= ATB[e]
            delta=gamma()* tmp
            xm = xm - delta
        delta = xm - x0
        check = 0
        while 1:
            lock1.acquire()
            if data_upd1 == 0:
                if data1==cores:
                    data1=0
                time.sleep(sleep_time)
                data1+=1
                if data1 ==1:
                    gldelta = 0
                gldelta += delta

                check = 1
                if data1 == cores:
                    data_upd1 = 1
            lock1.release()
            if check == 1 or conv == 1:
                break
        while 1:
            lock2.acquire()
            if data_upd2 == 1:
                if data2==cores:
                    data2=0
                xm = glxm
                data2 += 1
                check = 0
                if data2 == cores:
                    data_upd2 = 0
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
    global d
    global cores
    global L
    global x_init
    global start_time
    x2 = x_init
    x1 = x_init
    delta = np.zeros((d, 1))
    k = 0
    tmp = ''
    for i in range(cores):
        name = "#%s" % (i + 1)
        my_threads.append(threading.Thread(target=Slave, args=(name, i)))
        my_threads[i].start()
    g = open('t_arr.covtype.scale.txt', 'w')
    h = open('subopt_arr.covtype.scale.txt', 'w')
    g.truncate()
    h.truncate()

    start_time = time.time()
    while 1:
        tmp_cores = 0
        check = 0
        while 1:
            lock1.acquire()
            if data_upd1 == 1:
                delta = gldelta
                data_upd1 = 0
                check = 1
            lock1.release()
            if check == 1:
                x2 = x2 + delta
                break
        while 1:
            lock2.acquire()
            if data_upd2 == 0:
                glxm = x2
                data_upd2 = 1
                check = 0

            lock2.release()
            if check == 0:
                break
        print(x2)
        k = k + 1
        if (1 != conv):
            t_to_write = str(time.time() - start_time)
            g.write(t_to_write)
            g.write("\n")

            subopt_to_write = str(local_norm_2(x2 - x_star))
            h.write(subopt_to_write)
            h.write("\n")

        if is_conv(x1, x2) == 1:
            finish_time = time.time()
            print("It takes", finish_time-start_time)
            conv = 1
            print("lol")
            g.close()
            h.close()
            print("kek")
            break

        x1 = x2
    for i in range(cores):
        print('join')
        my_threads[i].join()
        print('lols')


if __name__ == "__main__":
    A = np.ones((d, d))
    A[4][5] = 100
    B = A
    B = A.T@B
    x_star = np.full((d, 1), 2)
    b = A@x_star
    L = max(abs(np.linalg.eig(np.matrix(B))[0]))
    print(L)
    ATB = A.T@b
    master()
