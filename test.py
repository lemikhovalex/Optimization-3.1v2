import random
import time
import numpy as np
import scipy.optimize as opt
import math
import threading
n = 500 #581012
d = 54
data_upd1 = 0
data_upd2 = 0
lock1 = threading.Lock()  # общение с мастером
lock2 = threading.Lock()
data1 = ''
data2 = ''
gldelta=np.zeros((d+1, 1))
glxm=np.zeros((d+1, 1))
testcounter = 0
conv = 0
epsilon_conv = 10 ** -9
L = 21930585.25*10**-8
p = 1
lambda_1 =10 ** -6
cores = 5 #для вычислений, один на главный


def scal_mul(x, z):
    global d
    out = 0
    for i in range(d):
        out += x[i+1]*z[i+1]
    return out




def nabla_z_l_j(x, z_j):
    global d
    global n
    out = np.zeros((d+1, 1))
    out[0][0] = 0
    for i in range(1, d+1):
        out[i] = 1.0
        out[i] /= (1 + math.exp(z_j[0][0] * scal_mul(x, z_j)))
        out[i] *= (-z_j[i] * z_j[0][0])
    return out


def reading_dataset(file_name):
    global n
    global d
    out = np.zeros((n, d+1))
    f = open(file_name, 'r')
    for i in range(n):
        s = f.readline()
        tokens = s.split(' ')
        out[i][0] = int(tokens[0])
        out[i][0] = - 3 + 2 * out[i][0]
        del tokens[0]
        while len(tokens) > 1:
            num_and_val = tokens[0].split(':')
            out[i][int(num_and_val[0])] = float(num_and_val[1])
            del tokens[0]
    return out


def local_norm_2(x):
    global d
    out = 0
    for i in range(1, d+1):
        out += x[i][0]**2
    return math.sqrt(out)


def local_norm_1(x):
    global d
    out = 0
    for i in range(1, d+1):
        out += math.fabs(x[i][0])
    return out


def gamma():
    global L
    return 1.0/L


def is_conv(x_curr, x_prev):
    delta = local_norm_2(x_curr - x_prev)
    if delta < epsilon_conv:
        return True
    else:
        return False


def r(x):
    global lambda_1
    global n
    out = 0
    for i in range(1, len(x)):
        out += x[i]**2
    out *= 1/(2*n)
    for i in range(1, len(x)):
        out += math.fabs(x[i])*lambda_1
    return out


def f_for_prox_gr(z, x):
    out = 0
    out += 1/(2*gamma())*local_norm_2(x)**2
    out += r(z)
    return out


def n_i(slave_num):
    global n
    global cores
    if (slave_num == 0):
        return 0
    if (n%cores == 0):
        out = int(n/cores)
    else:
        if(slave_num == cores):
            out = n - int(n/cores) * (cores - 1)
        else:
            out = int(n/cores)
    return out


def pi(slave_num):
    global n
    return n_i(slave_num)/n


def prox(x):
    global d
    res = opt.minimize(
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
    out = np.zeros((d+1, 1))
    for i in range(0, d+1):
        out[i] = res.x[i]
    out[0] = x[0]

    return out


def sub_in_xplus(slave_num, x):
    #xplus = z - sub_in_xplus(slave_num, z)
    global A
    global d
    out = np.zeros((d+1, 1))
    z_j = np.zeros((d+1, 1))
    for j in range( (slave_num - 1) * n_i(slave_num-1), (slave_num - 1) * n_i(slave_num-1) + n_i(slave_num) ):
        for i in range(d+1):
            z_j[i] = A[j][i]
        out += nabla_z_l_j(x, z_j)
    out *= gamma()
    out /= n_i(slave_num)
    return out


def Slave(name, num):
    global A
    global d
    xm = np.ones((d+1, 1))
    x = np.ones((d+1, 1))
    for i in range(1, d + 1):
        xm[i] = 20
        x[i] = 20
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
    print(pi(num))
    while 1:
        delta = np.zeros((d + 1, 1))
        for i in range(p):
            z = prox(xm + delta)

            subs = sub_in_xplus(num, z)
            print(num)
            print('я смог1')
            xplus = z - subs
            print(num)
            print('я смог2')
            delta += (xplus - x) * pi(num)
            print(num)
            print('я смог3')
            x = xplus
        check = 0
        while 1:
            lock1.acquire()
            if data_upd1 == 0:
                data1 = name
                gldelta = delta
                data_upd1 = 1
                check = 1

            lock1.release()
            if check == 1 or conv == 1:
                break
        while 1:
            lock2.acquire()
            if data_upd2 == 1 and name == data2:
                xm = glxm
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
    global d
    global cores
    global L
    x2 = np.ones((d+1, 1))
    x1 = np.ones((d+1, 1))
    for i in range(1, d+1):
        x1[i] = 20
        x2[i] = 20
    delta = np.zeros((d+1, 1))
    k = 0
    tmp = ''
    for i in range(cores):
        name = "Slave #%s" % (i + 1)
        my_threads.append(threading.Thread(target = Slave, args = (name, i+1)))
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
                x2 = x2 + delta
                break

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
        print(k)
        L*=2

        if is_conv(x1, x2) == 1:
            conv = 1
            break
        x1 = x2
        print(x1)
    for i in range(cores):
        print('join')
        my_threads[i].join()
        print('lols')


if __name__ == "__main__":
    A = reading_dataset("covtype.libsvm.binary.scale")
    master()
