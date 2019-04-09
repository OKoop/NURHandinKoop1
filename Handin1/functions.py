import numpy as np
from matplotlib import pyplot as plt
import sys
#Furthermore I assumed that, if we can use np.exp we can use np.log, and I use
#linspace/logspace because these are so easy to implement I wanted to save that
#until the end for if I have time left.
#For finding minimal and maximal values from a vector/list I use the standard
#max() and min() from python, for these are not from special libraries.


#Function that returns the log of the factorial (log(x!)) of an integer x. I
#assumed that we could use np.log if we can use np.exp, if not, sorry.
def logfact(n):
    s = 0
    for i in range(1, n+1):
        s = np.float64(s + np.log(i))
    return s

#Function that returns P_{\lambda}(k) for given \lambda and k, where P is the
#Poisson distribution. Makes sure in every step that the memory limit is not 
#violated. \lambda is replaced by 'l' for brevity.
def Poisson(l, k):
    l, k = np.float64(l), np.int64(k)
    
    #calculate l^k and e^(-l)
    A = np.float64(l**k)
    B = np.float64(np.exp(-l))
    
    #use logfact to be able to safely find k! by using logs and exponents.
    C = np.float64(np.exp(logfact(k)))
    
    #return the answer E
    E = np.float64((A * B)/C)
    return E

#Defining the Random Number Generator used in this exercise. This one is based
#on the lectures and values from there. It implements the Ranq1-generator from
#'Numerical Recipes'
class RNG(object):
    
    #Initialize the LCG-parameters and XOR-parameters and the state equal to 
    #the seed.
    def __init__(self, seed, a=1664525, c=1013904223, m=2**32, x1=21, x2=35, 
                 x3=4):
        self.state = np.int64(seed)
        self.a = a
        self.c = c
        self.m = m
        self.x1, self.x2, self.x3 = x1, x2, x3
    
    #A linear congruential generator, replacing the state with the next value
    #Based on slides and 'Numerical Recipes'
    def LCG(self):
        self.state = np.int64((self.a * self.state + self.c) % self.m)
    
    #An XOR-shift, replacing the state with the next value
    #Based on slides and 'Numerical Recipes'
    def XOR(self):
        x = self.state
        x ^= x >> self.x1
        x ^= x << self.x2
        x ^= x >> self.x3
        self.state = np.int64(x)
    
    #Function that returns one random number
    def randn(self):
        self.LCG()
        self.XOR()
        return self.state
    
    #Function that returns a random number or a sample of size 'size'
    #in a given interval given by [mini, maxi]
    def sample(self, size, mini=0, maxi=1):
        if size==1:
            return (self.randn()/sys.maxsize) * (maxi-mini) + mini
        else:
            samp = np.zeros(size, dtype=np.uint64)
            for i in range(size):
                samp[i] = self.randn()
            samp = samp/sys.maxsize
            return samp * (maxi - mini) + mini
    
    #Function specifically implemented to sample radii from p(x) as defined in
    #the assignment.    
    def genranrad(self, size, a, b, c, A, mrad, ma):
        samp = np.zeros(size)
        for i in range(size):
            probxg = 0
            rand = 1
            
            #We'll use rejection sampling, generating a radius x between 0 and
            #'mrad', then generating a random number f between 0 and max(p(x)).
            #If then f<p(x) we accept the radius x.
            while probxg <= rand:
                xg = self.sample(1, maxi=mrad)
                probxg = p(xg, a, b, c, A)
                rand = self.sample(1, maxi=ma)
            samp[i] = xg
        return samp
    
    #Returns the state.
    def state(self):
        return self.state

#Plots a simple scatterplot and saves it.
def scatter(x, y, t, xl, yl, svn=' ', sv=False, loglog=False):
    plt.scatter(x,y)
    plt.title(t, fontsize=18)
    plt.xlabel(xl, fontsize=15)
    plt.ylabel(yl, fontsize=15)
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    if sv:
        plt.savefig(svn+'.pdf', format='pdf')
    plt.close()
    return

#Plots a simple plot and saves it.
def plot(x, y, t, xl, yl, svn=' ', sv=False, loglog=False):
    plt.plot(x, y)
    plt.title(t, fontsize=18)
    plt.xlabel(xl, fontsize=15)
    plt.ylabel(yl, fontsize=15)
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    if sv:
        plt.savefig(svn+'.pdf', format='pdf')
    plt.close()
    return

#plots a simple histogram (density) and saves it.
def plthist(x, bina, t, xl, yl, svn=' ', sv=False):
    plt.hist(x, bins=bina, density=True)
    plt.title(t, fontsize=18)
    plt.xlabel(xl, fontsize=15)
    plt.ylabel(yl, fontsize=15)
    if sv:
        plt.savefig(svn+'.pdf', format='pdf')
    plt.close()
    return

#The function n(x) from the Hand-in set. We set A=1 by default to calculate A
#when given a, b, and c.
def n(x, a, b, c, Nsat, A=1):
    B = (x/b)**(a - 3.)
    C = np.exp(-(x/b)**c)
    return A * Nsat * B * C

#The function n(x)*x^2 which should be used for integrating n(x) in spherical
#coordinates.
def nspher(x, a, b, c, Nsat, A=1):
    B = (x/b)**(a - 1.)
    C = np.exp(-(x/b)**c)
    return A * Nsat * B * C * b**2.

#Romberg-algorithm for integration.
#We use this algorithm without any extra methods. It is good enough as is, and
#the order is manageable. It is the best method discussed in lecture 
#(in my opinion)
def romberg(func,a,b,order,args=[]):
    #Start by creating an array to save values
    res = np.zeros((order,order), dtype=np.float64)
    #Create an array with powers of 4 to be used.
    pow_4 = 4**np.arange(order)
    
    #The first 'interval size' is just the size of the starting interval
    h = (b-a)
    #The first value is given by h*(mean of f(a) and f(b))
    res[0,0] = h*(func(a, *args) + func(b, *args))/2.
    
    #Per step decrease the interval size and update the array with results by
    #combining previously known values and using the new points and thus values
    for i in range(1,order):
        h /= 2.
        
        res[i,0] = res[i-1,0]/2.
        #Including new values at only the new points.
        res[i,0] += h*sum(func(a+j*h, *args) for j in range(1,2**i+1,2))
        
        #Update new results as specified by the formula in Lecture 3 
        #(as analogue to Neville's algorithm).
        for k in range(1, i+1):
            res[i,k] = (pow_4[k]*res[i,k-1] - res[i-1,k-1])/(pow_4[k]-1)
    
    return res[-1,-1]

#Gives the y-value for a specific x on a line between two given points.
def line(x1, x2, y1, y2, x):
    a = (y2 - y1)/(x2 - x1)
    b = y1 - a * x1
    return a * x + b

#Linear Interpolation algorithm that also checks for possible extrapolation
#I've chosen for linear interpolation because it does the best job for all
#points below 1. The other algorithms do not improve the performance on higher
#points enough to compensate for the lack of performance on the lower points.
#Interpolation is done in log-space because then the performance is great on a
#bigger amount of points.
def linint(xs, ys, xval):
    i1 = np.abs(xs - xval).argmin()
    i2 = xs.argmax()
    i3 = xs.argmin()
    #Check if we need to extrapolate possibly
    if xval>max(xs):
        return line(xs[i2-1], xs[i2], ys[i2-1], ys[i2], xval)
    elif xval<min(xs):
        return line(xs[i3], xs[i3+1], ys[i3], ys[i3+1], xval)
    #Check where the x to interpolate on lies and interpolate.
    elif xs[i1]-xval<0:
        return line(xs[i1], xs[i1+1], ys[i1], ys[i1+1], xval)
    else:
        return line(xs[i1-1], xs[i1], ys[i1-1], ys[i1], xval)

#Function to determine central difference for derivatives in one point
def ysh(xs, func, h, args=[]):
    A = func(xs + h, *args) - func(xs - h, *args)
    ys = float(A/(2. * h))
    return ys

#Implementation of Ridders method based on the Romberg method, with terminology
#From the slides. This is the algorithm of which I expect the best result. Also
#it was easiest to implement for I already had Romberg implemented above.
def Ridders(func, xs, hstart, d, m, args=[]):
    results = [[0.0 for i in range(m)] for j in range(m)]
    ds = d**np.arange(m)
    
    h=hstart
    
    results[0][0] = ysh(xs, func, h, args)
    
    for i in range(1, m):
        h /= 2.
        
        results[i][0] = ysh(xs, func, h, args)
        
        for k in range(1, i + 1):
            results[i][k] = ((ds[k] * results[i][k-1] - 
                             results[i-1][k-1])/(ds[k] - 1))
    
    return results[-1][-1]

#The analytical derivative of n(x), calculated by hand and to check the 
#numerical value.
def ndiff(x, a, b, c, Nsat, A):
    B = np.exp(-(x/b)**c)
    C = (a - 3.)/b
    D = (x/b)**(a - 4)
    E = (x/b)**(a - 4 + c)
    F = c/b
    return A * Nsat * B * ( C * D - E * F)

#The p(x) as defined in the assignment
def p(x, a, b, c, A):
    return n(x, a, b, c, 1., A) * 4 * np.pi * x**2.

#-p(x), to be used for minimization.
def mp(x, a, b, c, A):
    return -n(x, a, b, c, 1., A) * 4 * np.pi * x**2.

#Function to merge two sorted arrays to be used in mergesort below.
def merge(left, right):
    result = []
    l, r = 0, 0
    lenl, lenr = len(left), len(right)
    #Iterate through both left and right and append the lowest value to 
    #'result'
    while l < lenl and r < lenr:
        if left[l] <= right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    #Append the 'residue' of the list that was not completely appended yet.
    result.extend(left[l:])
    result.extend(right[r:])
    return result

#Mergesort algorithm for sorting an array. I implemented mergesort because it
#is quite fast, and was easy to implement.
def mergesort(a):
    N = len(a)
    #If not already 1 element long, split the array and sort both parts. This
    #is done recursively.
    if N > 1:
        midi = N//2
        left, right = a[:midi], a[midi:]
        left, right = mergesort(left), mergesort(right)
        a = merge(left, right)
    return a

#The Golden Section
phi = (1. + 5. ** 0.5) / 2.

#Algorithm that implements the Golden Section Search as explained in lecture.
#This is used in Brents algorithm.
def GoldenSectSearch(func, a, b, taracc, maxit, args=[]):
    #find the function values.
    ya, yb = func(a, *args), func(b, *args)
    #if the function values are not as needed, change b and a.
    if ya < yb:
        a, b = b, a
    
    #do a golden section step, both ways.
    c = b - (b - a) / phi
    d = a + (b - a) / phi 
    i = 0
    
    acc = abs(b - a)
    
    #as long as the target accuracy or maximum iterations is not reached:
    #update using Golden Section steps.
    while acc > taracc and i<maxit:
        if func(c, *args) < func(d, *args):
            b = d
        else:
            a = c
        
        #After updating the limits start all over again before the next step.
        ya, yb = func(a, *args), func(b, *args)
        if ya < yb:
            a, b = b, a
        
        acc = abs(b - a)
        
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        i += 1.

    return ((b + a) / 2., acc, i)

#Performs Brents algorithm for finding the minimum of a function
def Brent(func, a, c, taracc, maxit, args=[]):
    #Give the first accuracy
    if c!=0:
        acc = np.abs((c-a)/c)
    elif a!=0:
        acc = np.abs((c-a)/a)
    else:
        print('all limits are 0')
        return
    i = 0
    #the first guess for the minimum assuming it was bracketed.
    b = (c+a)/2.
    
    while acc>taracc and i<maxit:
        #Calculate the new argmin by inverse parabolic interpolation
        R = func(b, *args)/func(c, *args)
        S = func(b, *args)/func(a, *args)
        T = func(a, *args)/func(c, *args)
        P = S*(T*(R-T)*(c-b)-(1-R)*(b-a))
        Q = (T-1.)*(R-1.)*(S-1.)
        d = b+P/Q
        #Check how to update the values used for bookkeeping 
        if d>b:
            cn = c
            an = b
        else:
            an = a
            cn = b
        bn = d
        #Perform a section step if the Brent point is not between the wanted 
        #interval. Or when the step is too big.
        if d<a or d>c or cn-an>(c-a)/2.:
            prodl = func(a, *args) * func(b, *args)
            check = func(b, *args)
            prodr = check * func(c, *args)
            if check==0:
                print('root '+str(b)+' found in '+str(i)+' iterations')
                return b
            elif prodl<0:
                c = b
            elif prodr<0:
                a = b
            b = (a+c)/2.
        else:
            a = an
            b = bn
            c = cn
        #Update the accuracy
        if c!=0:
            acc = np.abs((c-a)/c)
        elif a!=0:
            acc = np.abs((c-a)/a)
        else:
            print('all limits are 0')
            return
        i += 1
    return(b,acc,i)

#Function to find a normalization constant A (to normalize to 100.)
#given all other arguments. It is assumed the integral is spherical and
#independent of \phi and \theta.
def calcA(func, a, b, order, args=[], result=100.):
    #Find the integral of the function.
    rint = romberg(func, a, b, order, args)
    intedA = rint * 4 * np.pi
    #Return thus the 'normalization' constant
    return result/intedA

#The class that defines the interpolator for A given a, b and c. It uses linear 
#interpolation.
class interpolator(object):
    
    #Initialize the table with values of A given ranges of a, b and c.
    def __init__(self, func, ranges, order):
        self.rana = np.arange(ranges[0][0], ranges[0][1] + .1, 0.1)
        self.ranb = np.arange(ranges[1][0], ranges[1][1] + .1, 0.1)
        self.ranc = np.arange(ranges[2][0], ranges[2][1] + .1, 0.1)
        la, lb, lc = len(self.rana), len(self.ranb), len(self.ranc)
        self.table = np.zeros((la, lb, lc))
        #for each a, b and c find A
        for i in range(la):
            for j in range(lb):
                for k in range(lc):
                    self.table[i][j][k] = calcA(func, 0, 5, order, 
                                                [self.rana[i], self.ranb[j], 
                                                 self.ranc[k], 100])
    
    #Function to print the table.
    def p(self):
        print(self.table)
        
    #Function to locate the values of a, b or c closest below the value of a, b
    #or c that we want to know A for. m here is a number that tells how many
    #points we need to find the interpolated value, and is defined by the
    #algorithm used.
    def locate(self, r, n, m):
        if r=='a':
            ran = self.rana
        if r=='b':
            ran = self.ranb
        if r=='c':
            ran = self.ranc
        
        #Enclose the value from above and below until the pointers cross. This
        #method was sort of copied from Numerical Recipes.
        il, ih = 0, len(ran) - 1
        while (ih > 1 + il):
            im = (ih + il)//2
            if n >= ran[im]:
                il = im
            else:
                ih = im
        return max(0, min(len(ran)-m, il - (m - 2)//2))
    
    #Function to find the indices with 'locate' for a, b and c to use for
    #interpolation.
    def inds(self, a, b, c, m):
        return self.locate('a', a, m), self.locate('b', b, m), self.locate('c', 
                                                                            c, 
                                                                            m)
    
    #Linear interpolation in three dimensions, assuming the indices given are
    #those closest below the value we want to know, and we can not extrapolate
    #This was chosen because I did not yet have time to check if I could do
    #this with other methods. This was the easiest for now.
    def linearA(self, a, b, c, m):
        ailo, bilo, cilo = self.inds(a, b, c, 2)
        rana, ranb, ranc = self.rana, self.ranb, self.ranc
        tab = self.table
        
        ad = (a - rana[ailo])/(rana[ailo+1] - rana[ailo])
        bd = (b - ranb[bilo])/(ranb[bilo+1] - ranb[bilo])
        cd = (c - ranc[cilo])/(ranc[cilo+1] - ranc[cilo])
        
        c00 = tab[ailo][bilo][cilo] * (1 - ad) + tab[ailo+1][bilo][cilo] * ad
        c01 = (tab[ailo][bilo][cilo+1] * (1 - ad) + tab[ailo+1][bilo][cilo+1] * 
               ad)
        c10 = (tab[ailo][bilo+1][cilo] * (1 - ad) + tab[ailo+1][bilo+1][cilo] *
               ad)
        c11 = (tab[ailo][bilo+1][cilo+1] * (1 - ad) + 
               tab[ailo+1][bilo+1][cilo+1] * ad)
        
        c0 = c00 * (1 - bd) + c10 * bd
        c1 = c01 * (1 - bd) + c11 * bd
        
        return c0 * (1 - cd) + c1 * cd
    
#Note: a, b, c should be 2D-vectors.
def order(func, a, b, c):
    fa, fb, fc = func(a), func(b), func(c)
    if fa <= fb <= fc:
        return a, b, c
    elif fb <= fa <= fc:
        return b, a, c
    elif fb <= fc <= fa:
        return b, c, a
    elif fa <= fc <= fb:
        return a, c, b
    elif fc <= fa <= fb:
        return c, a, b
    else:
        return c, b, a

#An attempt at making a Downhill Simplex Algorithm
def DHS(func, start, taracc, maxit):
    x1 = np.array([start[0], start[1]+.1])
    x2 = np.array([start[0]+.1, start[1]])
    x3 = np.array([start[0]-.1, start[1]-.1])
    
    x1, x2, x3 = order(func, x1, x2, x3)
    cent = (x1 + x2 + x3)/3.
    print(x1, x2, x3)
    
    i=0
    
    while (abs(func(x3)-func(x1))/(abs(func(x3)+func(x1))/2.)) > taracc and i < maxit:
        xtry = 2. * cent - x3
        
        f1, ftry, f3 = func(x1), func(xtry), func(x3)
        if f1 <= ftry < f3:
            x3 = xtry
        elif ftry < f1:
            xtry2 = 2. * xtry - cent
            if func(xtry2) < f1:
                x3 = xtry2
            else:
                x3 = xtry
        else:
            xtry = (cent + x3)/2.
            if func(xtry) < f3:
                x3 = xtry
            else:
                x2 = (cent + x2)/2.
                x3 = (cent + x3)/2.
                
        x1, x2, x3 = order(func, x1, x2, x3)
        cent = (x1 + x2 + x3)/3.
        i += 1
        
    return x1, i, abs(func(x3)-func(x1))/(abs(func(x3)+func(x1))/2.)

#To be used in the conjugate gradient method
def functolinmin(a, func, xi, ni):
    return func((xi + a * ni))

#An implementation of the conjugate gradient method. This can not be used, for
#finding the gradient of the log-likelihood is a pain in the ***.
def conjgrad(func, gradfunc, start, tal=10**-6, taracc=10**-5., maxit=100):
    i = 1
    ni = -gradfunc(*start)
    nim1 = ni
    gi = ni
    gim1 = ni
    acc = (sum(abs(ni)**2.))**.5
    acc2 = acc
    a = GoldenSectSearch(functolinmin, 0, 100, tal, 50, args=[func, start, ni])[0]
    xi = start + a * ni
    while acc > taracc and acc2 > taracc and i < maxit:
        gtemp = -gradfunc(*xi)
        gim1 = gi
        gi = gtemp
        gammi = ((gi - gim1) @ gi)/(gim1 @ gim1)
        ntemp = gi + gammi * nim1
        nim1 = ni
        ni = ntemp
        a = GoldenSectSearch(functolinmin, a-1/(2**i), a+1/(2**i), tal, 50, args=[func, xi, ni])[0]
        xtemp = xi + a * ni
        acc = abs(func(*xtemp)-func(*xi))/(abs(func(*xtemp)+func(*xi))/2)
        acc2 = abs(func(*xtemp)-func(*xi))
        xi = xtemp
        i += 1
    return(xi, a, acc, acc2, i)

seed = 39509245
print('The seed value is equal to:',seed,'\n')
RNG1 = RNG(seed)