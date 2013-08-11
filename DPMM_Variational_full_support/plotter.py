#tejask@mit.edu

from pylab import imread,imshow,figure,show,subplot
from numpy import reshape,uint8,flipud
from scipy.cluster.vq import kmeans,vq
import numpy
import scipy.misc
import Image
import copy
import glob
import pdb
import pylab
import pickle
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import math


def saveAsPDF(fname,plot):
    pp = PdfPages(fname)
    pp.savefig(plot)
    pp.close()


def genericPlot(X,Y,xlab,ylab,fname):
    f = pylab.figure()
    ax=f.add_subplot(111,title='')
    pyplot.plot( X,Y,'-',color='blue', linewidth=2)
    pyplot.xlabel(xlab,fontsize=30)
    pyplot.ylabel(ylab,fontsize=30)
    pylab.savefig(fname+'.png')
    #ax.grid(True)
    saveAsPDF(fname+'.pdf',f)




fname = 'aug11_13/TRIALS_100_PARTICLES_0_DELTA_50'
data = pickle.load(open(fname+".pkl","rb"))

f = pylab.figure()
ax=f.add_subplot(111,title='')
X=[]
Y=[]
CNT=0
nolookArr =[]
lookArr = []
for i in range(len(data)):
    if len(data[i]) > 0:
        nolook = float(data[i].split('\n')[0].replace("[","").replace("]","").split(",")[3])
        look = float(data[i].split('\n')[0].replace("[","").replace("]","").split(",")[4])
        nolookArr.append(nolook)
        lookArr.append(look)
        X.append(i)
        Y.append(look-nolook)
        if look > nolook:
            CNT+=1

print len(data), CNT
print 'Average (Look):', sum(lookArr)/len(data)
print 'Average (Nolook):', sum(nolookArr)/len(data)

ax.bar(X,Y,0.4,color='black')

pylab.xlabel('Run Number',fontsize=30)
pylab.ylabel('VSCORE Difference',fontsize=30)
pylab.savefig(fname+'.png')
#pylab.ylim([-0.35, 0.35])
#ax.grid(True)
saveAsPDF(fname+'.pdf',f)


