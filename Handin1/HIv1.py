import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import functions as ft

#------------Exercise 1(a)---------------------
print('\n Output for Exercise 1(a):')
lk = [(1.,0), (5.,10), (3.,21), (2.6,40)]
for x in lk:
    print('For l='+str(x[0])+' and k='+str(x[1])+', Pl(k)='+ 
          '{:.6g}'.format(ft.Poisson(x[0],x[1]))+'.')



#------------Exercise 1(b)---------------------
print('\n Output for Exercise 1(b):')
#create a random sample of (hopefully) uniform numbers between 0 and 1.
samp = ft.RNG1.sample(1000000)
samp1000 = samp[0:1000:1]
sshift = samp[1:1001:1]
#find the scatter of the values
ft.scatter(sshift, samp1000, 'Sequential Random Numbers', r'$x_{i+1}$', r'$x_i$'
        ,'scatter1b',True)
#plot the histogram of the sample.
ft.plthist(samp, 20, 'Distribution of the RNG', 'Result', 'Probability','hist1b',
        True)


#------------Exercise 2(a)---------------------
print('\n Output for Exercise 2(a):')
Three = ft.RNG1.sample(3)
a = Three[0] * 1.4 + 1.1 
b = Three[1] * 1.5 + .5
c = Three[2] * 2.5 + 1.5 

#Luckily this integral splits into a product of three integrals, so we do
#not have to perform intricate integrations. 
rint = ft.romberg(ft.nspher, 0, 5, 15, args=[a, b, c, 100])
intedA = rint*4*np.pi
A = 100./intedA
print('For a='+str(a)+', b='+str(b)+', and c='+str(c)+':')
print('The normalization factor A='+str(A)+'.')

#------------Exercise 2(b)---------------------
print('\n Output for Exercise 2(b):')
#We are going to integrate in the log-log space as explained above.
#These are the given values.
xs = [10**-4., 10**-2., 10**-1., 1, 5]
nxs = ft.n(xs,a,b,c,100,A)

#Define a range in logspace and find the function values.
lxvals = np.linspace(-4, np.log10(5), 10000)
lxs = np.log10(xs)
lnxs = np.log10(nxs)

#Find the linearly interpolated values (in logspace) for n(x).
lyvals = []
for j in lxvals:
    lyvals.append(ft.linint(lxs, lnxs, j))
yvals = 10**np.array(lyvals)

xvals = 10**lxvals

#Plot the five points given and the interpolated lines. Also plot the original
#n(x).
plt.scatter(xs, nxs, c='r')
plt.plot(xvals, yvals, linestyle='dotted', c='g', linewidth = 3, 
         label='interpolated')
plt.plot(xvals, ft.n(xvals, a, b, c, 100, A), c='red', label='n(x)')
plt.xscale('log')
plt.yscale('log')
plt.title(r'$n(x)$ interpolated', fontsize=18)
plt.xlabel('x', fontsize=15)
plt.ylabel(r'$n(x)$', fontsize=15)
plt.xlim([10**-4., 5.])
plt.legend(loc=3, fontsize=15)
plt.savefig('2b.pdf', format='pdf')
plt.close()
#------------Exercise 2(c)---------------------
print('\n Output for Exercise 2(c):')
#Using Ridders method to find the derivative, as well as evaluating the 
#analytical derivative, we see that the values are equal to quite high 
#precision.
dnum = float('{:.20g}'.format(ft.Ridders(ft.n, b, .1, 2., 15,
             args=[a, b, c, 100, A])))
dana = float('{:.20g}'.format(ft.ndiff(b, a, b, c, 100, A)))
print('The derivative in b='+str(b)+' is:\n'+str(dnum)+' numerically, and\n',
      str(dana)+' analytically.')

#------------Exercise 2(d)---------------------
print('\n Output for Exercise 2(d):')
#Creating random sequences of radii, phi and theta and combining them in one list
#to save as csv file.

#First find the maximum of p(x)
mininf = ft.GoldenSectSearch(ft.mp, .1, 1.5, 10**-10., 100, [a, b, c, A])
ma = ft.p(mininf[0], a, b, c, A)

#Generate a random sample of radii, phis and thetas
randradiid = ft.RNG1.genranrad(100, a, b, c, A, 5., ma)
phis = 2 * np.pi * ft.RNG1.sample(100)
thetas = np.arccos(1. - 2. * ft.RNG1.sample(100))

#Save the samples to a file.
listsd = list(zip(randradiid, phis, thetas))
df = pd.DataFrame(listsd, columns=["r","phi","theta"])
df.to_csv('2d.csv', index=False)

#Find the bin in which the maximal amount of elements reside
def f(x):
    return hist[x]
binedges = np.logspace(-4, np.log10(5), 21)
hist, edges = np.histogram(randradiid, bins=binedges)
istat = max(np.arange(len(hist)), key=f)

#------------Exercise 2(e)---------------------
print('\n Output for Exercise 2(e):')
#Save 1000 lists of parameters in a csv file (did not know how to separate them 
#clearly)
#All the while generating histograms of the values and averaging over those.
#(Did not know how to easily make this faster or create my own function for 
#this also should have been able to use plt.hist as well, so I used 
#np.histogram and to incorporate the task of 2(g) into this)
hist = np.zeros(20)
widths = [binedges[i+1] - binedges[i] for i in range(20)]
lists, maxnumber = [], []
for i in range(1000):
    #find random phis, psis and thetas.
    randradii = ft.RNG1.genranrad(100, a, b, c, A, 5., ma)
    phis = 2 * np.pi * ft.RNG1.sample(100)
    thetas = np.arccos(1. - 2. * ft.RNG1.sample(100))
    #create lists and histograms and find the amount of elements in the bin
    #which had the maximal amount of sattelites in my own sample.
    lists = list(zip(randradii, phis, thetas))
    histi = np.histogram(randradii, bins = binedges)[0]
    hist += histi
    maxnumber.append(histi[istat])
    #Save the lists to a file.
    df = pd.DataFrame(lists, columns=["r","phi","theta"])
    df.to_csv('2e.csv', index=False, mode='a')
hist /= 1000.

#Create a plot of N(x) and plot the histogram in the same plot.
xs = np.logspace(-4, np.log10(5), 100000)

#This is N(x):
ys = ft.p(xs, a, b, c, A) * 100.
plt.figure(figsize=(15,15))
plt.plot(xs, ys, c='r')
plt.bar(binedges[:-1], height = hist, width = widths, align='edge', fill=False)
plt.xscale('log')
plt.yscale('log')
plt.xlim([10**-4, 5])
plt.title('N(x) and the histogram of average amount of satellites', 
          fontsize=30)
plt.xlabel('x', fontsize=25)
plt.ylabel('N(x)', fontsize=25)
plt.savefig('2e.pdf', format='pdf')
plt.close()
#I am not happy with this figure, but I do not know why it looks so bad. I do
#not think it is due to the RNG, but it is not as if N(x) is defined wrongly.
#Therefore I think it is due to the rejection sampling method, but I did not
#have time to try to implement this in another way.


#------------Exercise 2(f)---------------------
print('\n Output for Exercise 2(f):')
#This is the function to find the root of to find the two solutions to N(x)=y/2
def rootpy2(x, a, b, c, A, maxN):
    return ft.p(x, a, b, c, A) * 100. - maxN/2.

#We already found the maximum of p(x), and only need to multiply this with 100
#to have the maximal value of N(x). 
maxN = ma * 100

r1 = ft.Brent(rootpy2, .2, 1, 10**-10., 100, [a, b, c, A, maxN])[0]
r2 = ft.Brent(rootpy2, 1.5, 2.5, 10**-10., 100, [a, b, c, A, maxN])[0]

print('The two solutions are: x='+str(r1)+' and x='+str(r2)+'.')


#------------Exercise 2(g)---------------------
print('\n Output for Exercise 2(g):')
#We want to find the list of amount of elements in the maximal bin and sort it,
#so we can find the percentiles.
eltsinmaxbin = []
for el in randradiid:
    if edges[istat] < el and el < edges[istat+1]:
        eltsinmaxbin.append(el)
sort = ft.mergesort(eltsinmaxbin)
N=len(sort)
print('16-th percentile: '+str(sort[int(N//6.25)])+', median: '+str(sort[N//2])+
      ', 84-th percentile: '+str(sort[int(N//(100/84.))])+'.')

#The lambda we find from the mean of the model, and k is the variable.
lamb = float(sum(maxnumber))/len(maxnumber)
binedges = np.arange(min(maxnumber), max(maxnumber)+1, 1)
ys = []
for i in binedges:
    ys.append(ft.Poisson(lamb, i))
ys = np.array(ys)
plt.hist(maxnumber, bins=binedges, density=True)
plt.plot(binedges,ys)
plt.xlabel(r'$\mathcal{P}(x)$', fontsize=15)
plt.ylabel('Amount of satellites', fontsize=15)
plt.title('Histogram and Poisson of amount of satellites', fontsize=18)
plt.savefig('2g.pdf', format='pdf')
plt.close()

#------------Exercise 2(h)---------------------
print('\n Output for Exercise 2(h):')
#Initialize the interpolator.
interpolAtor = ft.interpolator(ft.nspher, [(1.1, 2.5), (.5, 2), (1.5, 4)], 8)
#Calculate for the set a, b, and c- values to check the interpolator.
print('The interpolator gives:',interpolAtor.linearA(a, b, c, 2))
print('The true value is:',A)




#------------Exercise 3(a)---------------------
print('\n Output for Exercise 3(a):')

#-----------Exercise 3(b)-----------------
print('\n Output for Exercise 3(b):')






