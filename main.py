
from bisect import *
from email import header
from re import M
from subprocess import REALTIME_PRIORITY_CLASS
from tkinter import END
from turtle import width
import numpy as np
import math
import matplotlib.pyplot as plt
from bisect import *


def cwzad51():
    def f(x):
        y=x**4/10-2*x**2-x-3*np.sin(x)+5
        return y
    a=0
    b=2
    print('f({:.4f})={:.4f}'.format(a,f(a)))
    print('f({:.4f})={:.4f}'.format(b,f(b)))
    x=(a+b)/2.0
    print('f({:.4f})={:.4f}'.format(x,f(x)))

def cwzad52():
    def f(x):
        y = x ** 4 / 10 - 2 * x ** 2 + -x - 3 * np.sin(x) + 5
        return (y)


    tol = 0.0001

    a = 0

    b = 2
    # we will do the first iterate before our while loop starts so that we
    # have a value to test against the tolerance

    x = (a + b) / 2

    while np.abs(f(x)) > tol:

        print('a={:.5f} f(a)={:.5f}, b={:.5f} f(b)={:.5f}, 15 x={:.5f} f(x)={:.5f}'.format(a, f(a), b, f(b), x, f(x)))
        # now decide whether we replace a or b with x

        if f(a) * f(x) < 0:
            # root is between a and x so replace b

            b = x
        elif f(b) * f(x) < 0:
            # root is between b and x so replace a

            a = x
        else:

            # in this case, f(x) must be 0 and we have found the root
            break
        # recompute the approximation

        x = (a + b) / 2

    print('final x =', x)

    print('final f(x) =', f(x))
def shufted_exp(x):
    y=np.exp(3)-3
    return y

def bisect(f,a,b,tol):
    x=(a+b)/2
    while np.abs(f(x))>tol:
        print('a={:.5f} f(a)={:.5f}, b={:.5f} f(b)={:.5f}, x={:.5f} f(x)={:.5f}'.format(a, f(a), b, f(b), x, f(x)))

        if f(a)*f(x)<0:
            b=x
        elif f(b)*f(x)<0:
            a=x
        else:
            break
        x=(a+b)/2
    return x


def f(x):
    y = x ** 4 / 10 - 2 * x ** 2 - x - 3 * np.sin(x) + 5

    return y
def bisectzad1a(f,a,b,tol):

    x=(a+b)/2
    pom=0
    while np.abs(f(x))>tol:
        print('a={:.5f} f(a)={:.5f}, b={:.5f} f(b)={:.5f}, x={:.5f} f(x)={:.5f} \n'.format(a, f(a), b, f(b), x, f(x)))

        pom=pom+1
        if f(a)*f(x)<0:
            b=x
        elif f(b)*f(x)<0:
            a=x
        else:
            break
        pom=pom+1
        x=(a+b)/2
    return x,pom

# wyn=bisectzad1a(f,0,2,0.0001)
# print(wyn)
def zadaniepokazowe():
    def arangefromnumpy():
        t=np.arange(0,10,1)
        print(t)
        for i in t:
            print(i)

    # arangefromnumpy()

    def rhs(x):
        y=x**3-(100*np.cos(x))

        # Zwrócenie wektora pochodnych
        return y


    # Wartość początkowa dla t
    a = 1

    # Wartość końcowa dla t
    b = 4

    # Liczba podziałów przedziału czasowego
    n = 500

    # Obliczenie kroku czasowego
    dt = (b - a) / n

    # Stworzenie wektora wartości t
    t = np.arange(a, b + dt, dt)

    # Inicjalizacja macierzy dla wartości y
    y = np.zeros((n + 1, 2))

    # Inicjalizacja listy indeksów
    i = np.arange(1, n + 1, 1)

    # Warunki początkowe dla y
    y[0, 0] = 1.5
    y[0, 1] = 1

    # Pętla obliczeń dla metody Eulera
    for k in i:
        # Obliczenie przybliżenia metody Eulera
        dy = rhs(t[k - 1])

        # Obliczenie kolejnego przybliżenia
        y[k, :] = dy * dt + y[k - 1, :]

    # Rysowanie przybliżeń
    plt.plot(t, y[:, 0])
    #plt.plot(t, y[:, 1])

    # Ustawienia wykresu
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('t')
    plt.grid()
    plt.legend(['x(t)', 'y(t)'])
    plt.show()
def cwiczenie():
    def fun(x):
        y=x**2
        return y
    x=np.arange(0,10000,1)
    y=fun(x)
    plt.plot(x,y)
    plt.grid()
    #plt.legend() < - wymagane do pokazania legendy
    plt.show()

def bisectzad2a(f,a,b,tol):

    x=(a+b)/2
    prev_x=x
    pom=0
    while np.abs(f(x))>tol:
        print('a={:.5f} f(a)={:.5f}, b={:.5f} f(b)={:.5f}, x={:.5f} f(x)={:.5f} \n'.format(a, f(a), b, f(b), x, f(x)))

        pom=pom+1
        if f(a)*f(x)<0:
            b=x
        elif f(b)*f(x)<0:
            a=x
        else:
            break
        pom=pom+1
        x=(a+b)/2
        if np.abs(x - prev_x) < tol:  # Check absolute difference against tolerance
            break
        prev_x=x
    return x,pom
# wyn=bisectzad2a(f,0,2,0.000001)
# print(wyn)
#cwiczenie()
def zad3cw():
    def fun(x,t,a):
        if x<=0 & x<3:
            y=x**t
        else:
            y=(a*t+3)/(t+5)
        return y
    x=np.arange(50,60)
    y=f(x)
    plt.plot(x,y)
    plt.grid()
    plt.show()
#zad3cw()
def zad3cw1():
    def f(x):
        y = np.exp(3) - (3 * x + 3) / 8
        return y

    x = np.arange(50, 60)
    y = f(x)
    plt.plot(x, y)
    plt.grid()
    plt.show()
    x, itcount = bisectzad1a(f, 50, 54, .0001)
    print('x = ', x)
    print('Number of iterations needed:', itcount)
    # check the answer
    print('e^3 =', np.exp(3))
    print('(3a+3)/(3+5) =', (x * 3 + 3) / (8))

#zad3cw1()

def zad4cw():
    # original function
    def f(x):
        y = np.exp(x) * np.sin(x) - x ** 2 / 2 + 5
        return y

    # derivative
    def df(x):
        y = np.exp(x) * (np.sin(x) + np.cos(x)) - x
        return y
    # plotting f' just to make sure there is a zero
    dx = 0.1
    x = np.arange(-1, 3 + dx, dx)
    y = df(x)
    plt.plot(x, y)
    plt.grid()
    plt.show()
    x, itcount = bisectzad1a(df, -1, 3, .0001)
    print('x = ', x)
    print('Number of iterations needed:', itcount)
    # check critical numbers and endpoints in the original function
    x_compare = [-1, 3, x]
    y_compare = [f(-1), f(3), f(x)]
    # determine which index holds the max value
    max_i = np.argmax(y_compare)
    print('Max of {:.4f} occurs at x={:.4f}.'.format(y_compare[max_i], x_compare[max_i]))
#zad4cw()
def zad5cw():
    # Newton's method
    # f is the original function
    # df is the derivative of f
    # x0 is the initial guess for the root

    def f(x):
        y = x ** 3 - 100 * np.cos(x)

        return y

    def df(x):
        y = 3 * x ** 2 + 100 * np.sin(x)

        return y
    def newtonroot(f, df, x0, tol):
        itcount = 0

        while np.abs(f(x0)) > tol:
            x1 = -f(x0) / df(x0) + x0
            itcount = itcount + 1
            x0 = x1
        return x0, itcount

    x, n = newtonroot(f, df, 1, .0001)
    print('x = ', x)
    print('Number of iterations needed:', n)


#zad5cw()
def zad6cw():
    def euler(rhs, a, b, y0, dt):

        # create a vector of t-values
        t = np.arange(a, b + dt, dt)
        n = len(t)
        # create space for the y-values
        y = np.zeros(n)
        # create a list of indices
        i = np.arange(1, n, 1)
        # we know the inital value of y to be 1
        y[0] = y0
        for k in i:
        # compute the Euler approximation
        # use the right hand side function to get the slope of the
        # tangent line
            m = rhs(t[k - 1], y[k - 1])
        # get the next approximation
            y[k] = m * dt + y[k - 1]
        return t, y

    def rhs(t, y):
        m = t ** 2 - np.sin(t)
        return m

    def trusol(x):
        y = x ** 3 / 3 + np.cos(x) - 1
        return y
    # Begin main program
    t, y = euler(rhs, 0, 2 * np.pi, 0, .1)
    plt.plot(t, y, 'b')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid()
    plt.plot(t, trusol(t), 'b--')
    plt.legend(['Approximation', 'True Solution'])
    plt.show()
#zad6cw()
def zad7cw():
    def euler(rhs, a, b, y0, dt):

        # create a vector of t-values
        t = np.arange(a, b + dt, dt)
        n = len(t)
        # create space for the y-values
        y = np.zeros(n)
        # create a list of indices
        i = np.arange(1, n, 1)
        # we know the inital value of y to be 1
        y[0] = y0
        for k in i:
        # compute the Euler approximation
        # use the right hand side function to get the slope of the
        # tangent line
            m = rhs(t[k - 1], y[k - 1])
        # get the next approximation
            y[k] = m * dt + y[k - 1]
        return t, y
    def rhs(t, y):
        r = np.cos(t) + np.exp(-t) * y
        return r
    a = 0
    b = 2 * np.pi
    dt = 0.05
    y0 = 0
    t, y = euler(rhs, a, b, y0, dt)
    plt.plot(t, y)
    # implicit Euler
    n = len(t)
    y2 = np.zeros(n)
    y2[0] = y0
    for i in range(1, n, 1):
        y2[i] = (np.cos(t[i]) * dt + y2[i - 1]) / (1 - np.exp(-t[i]) * dt)
    plt.plot(t, y2, 'b--')
    plt.legend(['Explicit Euler', 'Implicit Euler'])
    plt.show()
#zad7cw()
def zad8cw():
    def rhs(t, yvec):
        a = 0.04

        b = 0.0005
        c = -0.1
        d = 0.0005
        dy = np.zeros(2)
        dy[0] = a * yvec[0] - b * yvec[0] * yvec[1]
        dy[1] = c * yvec[1] + d * yvec[0] * yvec[1]
        return dy
    # initial t value
    a = 0
    # final t value
    b = 365
    # number of intervals
    n = 50000
    # delta t
    dt = (b - a) / n
    # create a vector of t-values
    t = np.arange(a, b + dt / 2, dt)
    # create space for the y-values
    y = np.zeros((n + 1, 2))
    # create a list of indices
    i = np.arange(1, n + 1, 1)
    y[0, 0] = 50
    y[0, 1] = 10
    for k in i:
        # compute the Euler approximation
        # use the right hand side function to get the slope of the tangent line
        dy = rhs(t[k - 1], y[k - 1, :])
        # get the next approximation
        y[k, :] = dy * dt + y[k - 1, :]
    # plot the approximations
    plt.plot(t, y[:, 0])
    plt.plot(t, y[:, 1])
    # plot true solution
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('t')
    plt.grid()
    plt.legend(['x(t)', 'y(t)'])
    plt.show()
    # --------------------------------------------------------
    # phase portrait
    plt.figure()
    plt.plot(y[:, 0], y[:, 1])
    head = 1
    tail = 0
    w = 55
    dx = y[head, 0] - y[tail, 0]
    dy = y[head, 1] - y[tail, 1]
    plt.arrow(y[head, 0], y[head, 1], dx, dy, width=.004)
    numarrows = int((n - head) / w)
    for i in range(4):
        head = head + w
        tail = tail + w
        dx = y[head, 0] - y[tail, 0]
        dy = y[head, 1] - y[tail, 1]
        plt.arrow(y[head, 0], y[head, 1], dx, dy, width=.004)
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.title('Phase Portrait: IC = X(0) = 50, Y(0) = 10')
    plt.grid()
    plt.show()
# zad8cw()
def zad9cw():
    def interp(x, t, y):
        n = len(t)

        startindex = 0
        # find the indices between which the new x value lies
        if x > np.max(t):
            print('outside of interpolation range')
            return np.nan
        elif x < np.min(t):
            print('outside of interpolation range')
            return np.nan
        while t[startindex] < x:
            startindex = startindex + 1
        startindex = startindex - 1
        endindex = startindex + 1
        # slope for interpolation
        m = (y[endindex] - y[startindex]) / (t[endindex] - t[startindex])
        # compute approximation using point slope form
        y_of_x = m * (x - t[startindex]) + y[startindex]
        return y_of_x
    t = np.array([0.000, 0.040, 0.080, 0.120, 0.160, \
                      0.200, 0.240, 0.280, 0.320, 0.360])
    y = np.array([1.000, 1.040, 1.078, 1.115, 1.150, \
                      1.182, 1.212, 1.240, 1.266, 1.289])
    new_t = np.linspace(0, .36, 76)
    n = len(new_t)
    new_y = np.zeros(n)
    for i in range(n):
        new_y[i] = interp(new_t[i], t, y)
    plt.plot(t, y, 'b-', alpha=.5)
    plt.plot(new_t, new_y, 'b-', alpha=.95)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()
zad9cw()