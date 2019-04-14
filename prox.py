
import math
import scipy
from scipy import optimize

epsilon_conv=10**-4
L=22*10**7
n=10000
lambda_1 =10**-5

#нормы считаются как будто нулевой элемент засран
def local_norm(x):
    out=0
    for i in range(1,len(x) ):
        out+=x[i]**2
    return math.sqrt(out)


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



def f_for_prox(z, x):
    out=0
    out+= 1/(2*gamma())*local_norm(x)**2
    out += r(z)
    return out


def prox_gr(x):
    res=scipy.optimize.minimize(
                            f_for_prox(),
                            (x),
                            args=(x),
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

