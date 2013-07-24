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




fname = 'result_10particles_30path'
data = pickle.load(open(fname+".pkl","rb"))

f = pylab.figure()
ax=f.add_subplot(111,title='')
X=[]
Y=[]
CNT=0

with_maxf=[]
without_maxf = []
with_eqmaxf = []

for i in range(len(data)):
    if len(data[i]) > 0:
        without_maxf.append(float(data[i].split('\n')[0].replace("[","").replace("]","").split(",")[0]))
        with_maxf.append(float(data[i].split('\n')[0].replace("[","").replace("]","").split(",")[1]))
        with_eqmaxf.append(float(data[i].split('\n')[0].replace("[","").replace("]","").split(",")[2]))
        X.append(i)

print 'Average (without_maxf):', sum(without_maxf)/len(data)
print 'Average (with_maxf):', sum(with_maxf)/len(data)
print 'Average (with_eqmaxf):', sum(with_eqmaxf)/len(data)


#ax.bar(X,Y,0.4,color='black')
ax.plot(X,without_maxf, color="grey")
ax.plot(X,with_maxf, color="black")
ax.plot(X,with_eqmaxf, color="blue")


pylab.xlabel('Run Number',fontsize=30)
pylab.ylabel('ARI Difference',fontsize=30)
pylab.savefig(fname+'.png')
#pylab.ylim([-0.35, 0.35])
#ax.grid(True)
saveAsPDF(fname+'.pdf',f)


