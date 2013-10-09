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
import operator

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




fname = 'results/K=1_putative_result_1particles_1path'
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
        if len(data[i])>1:
            print i, float(data[i].split('\n')[0].replace("[","").replace("]","").split(",")[0]), float(data[i].split('\n')[0].replace("[","").replace("]","").split(",")[1]), float(data[i].split('\n')[0].replace("[","").replace("]","").split(",")[2]) 
            without_maxf.append(float(data[i].split('\n')[0].replace("[","").replace("]","").split(",")[0]))
            with_maxf.append(float(data[i].split('\n')[0].replace("[","").replace("]","").split(",")[1]))
            with_eqmaxf.append(float(data[i].split('\n')[0].replace("[","").replace("]","").split(",")[2]))
            X.append(CNT);CNT+=1

print 'Average (without_maxf):', sum(without_maxf)/len(without_maxf)
print 'Average (with_maxf):', sum(with_maxf)/len(with_maxf)
print 'Average (with_eqmaxf):', sum(with_eqmaxf)/len(with_eqmaxf)


ax.bar(X,map(operator.sub, with_eqmaxf, with_maxf),0.05,color='black')

"""ax.plot(X,without_maxf, color="grey")
ax.plot(X,with_maxf, color="black")
ax.plot(X,with_eqmaxf, color="blue")"""


pylab.xlabel('Dataset',fontsize=35)
pylab.ylabel('V-Measure Diff',fontsize=35)# (30 samples avg/dataset)
pylab.savefig(fname+'.png')
#pylab.ylim([-0.35, 0.35])
#ax.grid(True)
saveAsPDF(fname+'.pdf',f)


