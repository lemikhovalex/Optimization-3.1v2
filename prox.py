
import math
import scipy
from scipy import optimize

cores=8
epsilon_conv=10**-4
L=22*10**7
n=10000
lambda_1 =10**-5
d=100

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

def local_norm_2(x):
    out=0
    for i in range(1,len(x) ):
        out+=x[i]**2
    return math.sqrt(out)

def local_norm_1(x):
    out=0
    for i in range(1,len(x) ):
        out+=math.fabs(x[i])
    return out

def gamma():
    global L
    return 1/L


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


def sub_in_xplus(slave_num, z_j, x):
    #xplus = z - sub_in_xplus(slave_num, z)
    global d
    out=[0]*(d+1)
    for j in range(n_i(slave_num)):
        out+=nabla_z_l_j(x, z_j)
    out*=gamma()
    out/=n_i(slave_num)
    return out
