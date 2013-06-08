#tejask@mit.edu 
from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pdb
from matplotlib.patches import Ellipse
import pylab
from matplotlib import pyplot
import math
import pickle
import datetime
import os, sys
import glob
from sklearn import metrics
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
import random,copy
import scipy.stats

def genericPlot(X,Y,xlab,ylab,fname):
    f = pylab.figure()
    ax=f.add_subplot(111,title='')
    pyplot.plot( X,Y,'-',color='blue', linewidth=2)
    pyplot.xlabel(xlab,fontsize=20)
    pyplot.ylabel(ylab,fontsize=20)
    ax.grid(True)
    pylab.savefig(fname)



class DPMixture():

    def __init__(self):
        self.state=dict()

    def plotClusters(self,mean,std,ax,col):
        for cluster in range(len(mean[0])):
            e=Ellipse(xy=numpy.array([mean[0][cluster], mean[1][cluster]]),width=3*std[0][cluster], height=3*std[1][cluster])#plotting 3 std's
            ax.add_artist(e)
            e.set_alpha(0.6)
            e.set_facecolor(col[cluster])


    def plotPoints(self,data,ax):
        parameters = data['parameters']
        for e in range(parameters['experiments']):
            for pt in range(parameters['n_per_experiment']):
                ax.plot(data[e][pt][0], data[e][pt][1],'o',color=(0,0,0))


    def get_data(self, dst,stdscale,_mean_, _std_):
        data = self.get_synthetic_data(dst=dst,stdscale=stdscale,_mean_=_mean_,_std_=_std_)
        pickle.dump(data,open(dst+"training_set_"+str(stdscale)+".pkl","wb"))
        pickle.dump(data,open("training_set_"+str(stdscale)+".pkl","wb"))
        """else:
            data = pickle.load(open("training_set.pkl","rb"))
            pickle.dump(data,open(dst+"/training_set.pkl","wb"))
        self.data=data"""
        return data


    def get_synthetic_data(self,dst,stdscale,_mean_="", _std_=""):
        parameters = {'experiments': 1, 'n_per_experiment': 100, 'dim': 2, 'max_clusters': NUM_CLUSTERS, 'mean_max': 10, 'stdscale':stdscale }
        data = dict()
        data['parameters'] = parameters
        true_clusters = []
        for e in range(parameters['experiments']):
            data[e] = dict()
            total_clusters = parameters['max_clusters']#numpy.random.randint(2,parameters['max_clusters'])
            cluster_col=numpy.random.rand(total_clusters,3)

            if _mean_ == "" and _std_ == "":
                mean = dict(); std=dict()
                for d in range(parameters['dim']):
                    mean[d] = numpy.zeros(total_clusters)
                    for jj in range(total_clusters):
                        mean[d][jj] = MyRIPL.sample("(- (uniform-continuous 0 20) 10)")
                    std[d] = numpy.zeros(total_clusters)
                    for jj in range(total_clusters):
                        #std[d][jj]=MyRIPL.sample("(+ (* (inv-gamma 1 "+str(parameters['stdscale'])+") (gamma 1 0.5)) 0.1)")
                        std[d][jj]=MyRIPL.sample("(+ (* (inv-gamma 1 "+str(parameters['stdscale'])+") 1) 0.1)")
                        #std[d][jj] = MyRIPL.sample("(+ (* (beta 1 10) 10) 0.10000000000000001)")
            else:
                mean = _mean_
                std = _std_

            if DEBUG_VIS == 1:
                fig = pylab.figure()
                ax = fig.add_subplot(111)

            for pt in range(parameters['n_per_experiment']):
                data[e][pt] = dict()
                cluster_id = numpy.random.randint(total_clusters)
                true_clusters.append(cluster_id)
                data[e][pt]['mean'] = dict()
                for d in range(parameters['dim']):
                    data[e][pt][d] = numpy.random.normal(mean[d][cluster_id], std[d][cluster_id])
                if DEBUG_VIS == 1:
                    ax.plot(data[e][pt][0], data[e][pt][1],'o', color=tuple(cluster_col[cluster_id]))
            if DEBUG_VIS == 1:
                self.plotClusters(mean,std,ax,cluster_col)
                ax.set_xlim(-15,15)
                ax.set_ylim(-15,15)
                ax.grid(True)
                pylab.savefig(dst+'/original'+str(stdscale)+'.png')
        data['true_clusters'] = true_clusters
        return data


    def sample_crp(self,n,alpha):
        clusterIDs=dict()
        clusterIDs[0] = 0

        def init_tables():
            tables = dict()
            tables[0] = 1
            return tables

        tables = init_tables()

        for i in range(1,n):
            occupied =  numpy.array([tables[t] for t in tables]) #tables[t] == number of customers sitting on table(t)
            new_table = alpha
            pdf = occupied
            pdf=abs(numpy.append(pdf,new_table))
            pdf = numpy.float64(pdf)/(i-1+alpha)
            pdf /= sum(pdf)
            sample = numpy.random.multinomial(1,pdf)
            sampled_cluster_id = numpy.where(sample > 0)[0][0]

            clusterIDs[i] = sampled_cluster_id
            tables = init_tables()
            for c in clusterIDs.values():
                tables[c] = 0
            for c in clusterIDs.values():
                tables[c] += 1
        clusterarr = numpy.unique(clusterIDs.values())
        return list(clusterarr), clusterIDs


    def get_alpha(self):
        alpha = 0.01
        return alpha

    def get_mean(self):
        return (numpy.random.rand()*-20) + 10 # mu ~ U(-10, 10)

    def get_std(self):
        return numpy.random.rand()*4 # std ~ U(0,4)
    
    def get_gaussin_loglikelihood(self, data, mu, std):
        epsilon = 0.00001
        loglikelihood = 0
        for d in range(2):
            loglikelihood += math.log(epsilon+scipy.stats.norm.pdf(data[d], mu[d], std[d]))
        return loglikelihood



    def prior(self):
        data = self.data; state = self.state; num_points = len(data)
        state[0] = dict()
        state[0]['c'] = dict(); state[0]['c_for_each_y'] = dict();
        state[0]['mu'] = dict(); state[0]['std'] = dict(); 

        state[0]['c'], state[0]['c_for_each_y'] = self.sample_crp(num_points,self.get_alpha())
        print state[0]['c']
        for c in state[0]['c']:
            state[0]['mu'][c] = [self.get_mean(), self.get_mean()]
            state[0]['std'][c] = [self.get_std(), self.get_std()]


    def draw_sample(self,pdf):
        pdf = numpy.exp(pdf); pdf /= sum(pdf);
        sample = numpy.random.multinomial(1,pdf)
        sample_ind = numpy.where(sample > 0)[0][0]
        return sample_ind


    def plotClusters(self,fname,state):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ## Plotting data points
        for pt in self.data:
            ax.plot(pt[0],pt[1],'o',color=(0,0,0))
        ## Plotting ellipses for clusters
        for key in state['mu'].keys():
            e=Ellipse(xy=numpy.array([state['mu'][key][0], state['mu'][key][1]]),width=3*state['std'][key][0], height=3*state['std'][key][1])#plotting 3 std's
            ax.add_artist(e)
            e.set_alpha(0.6)
            e.set_facecolor((numpy.random.rand(), numpy.random.rand(), numpy.random.rand()))
        ax.set_xlim(-15,15)
        ax.set_ylim(-15,15)
        ax.grid(True)
        pylab.savefig(fname+'.png')

    def chain(self):
        print 'MCMC Begins ...'
        data = self.data; state = self.state; 
        self.num_points = len(data)
        num_points = self.num_points

        for itr in range(1,100):
            epsilon = 0.00001
            print 'Iteration:', itr , ' ....'

            state[itr] = copy.deepcopy(state[itr-1])

            self.plotClusters(str(itr), state[itr])

            for i in range(len(data)):
                c_present = state[itr]['c_for_each_y'][i]
                n__i_c = len(numpy.where(numpy.array(state[itr]['c_for_each_y'].values()) == c_present)[0]) 
                if n__i_c <= 1:
                    del state[itr]['mu'][c_present]; del state[itr]['std'][c_present]

                c_vals = copy.deepcopy(state[itr]['c']) #list of all cluster IDs
                c_vals.append(max(c_vals)+1)

                #begin gibbs
                pdf = numpy.array([])
                tmpstate = copy.deepcopy(state)
                for cid in c_vals:
                    n__i_c = len(numpy.where(numpy.array(state[itr]['c_for_each_y'].values()) == cid)[0])
                    if n__i_c > 1:
                        obs_loglikelihood = self.get_gaussin_loglikelihood( data[i], state[itr]['mu'][cid], state[itr]['std'][cid] )
                        logL = math.log(n__i_c) + obs_loglikelihood - math.log(num_points - 1 + self.get_alpha())
                    else:
                        if state[itr]['mu'].has_key(cid) == False:
                            tmpstate[itr]['mu'][cid] =  [self.get_mean(), self.get_mean()]
                            tmpstate[itr]['std'][cid]=  [self.get_std(), self.get_std()]
                        obs_loglikelihood = self.get_gaussin_loglikelihood( data[i], tmpstate[itr]['mu'][cid], tmpstate[itr]['std'][cid] )
                        logL = math.log(self.get_alpha()) + obs_loglikelihood - math.log(num_points - 1 + self.get_alpha())
                    pdf = numpy.append(pdf, logL)


                pdf = numpy.exp(pdf);
                pdf += epsilon 
                pdf /= sum(pdf);
                
                sample = numpy.random.multinomial(1,pdf)
                c_sampled_ind = numpy.where(sample > 0)[0][0]
                c_sampled = c_vals[c_sampled_ind]

                if tmpstate[itr]['mu'].has_key(c_sampled): #new created
                    state[itr]['mu'][c_sampled] = copy.deepcopy(tmpstate[itr]['mu'][c_sampled])
                    state[itr]['std'][c_sampled] = copy.deepcopy(tmpstate[itr]['std'][c_sampled])

                state[itr]['c_for_each_y'][i] = c_sampled
                
                if (c_sampled in state[itr]['c']) == False:
                    state[itr]['c'].append(c_sampled)

                new_state = copy.deepcopy(state[itr]['c'])
                for ii in range(len(state[itr]['c'])):
                    if (state[itr]['c'][ii] in state[itr]['c_for_each_y'].values()) == False:
                        print 'DELETING:' , state[itr]['c'][ii]
                        del new_state[ii]

                state[itr]['c'] = new_state

                del tmpstate


            #sampling observation model
            mean_vals = []; std_vals = [];
            for ii in numpy.arange(-10,10,1):
                for jj in numpy.arange(-10,10,1):
                    mean_vals.append([ii, jj])

            for ii in numpy.arange(0.1,2,0.1):
                for jj in numpy.arange(0.1,2,0.1):
                    std_vals.append([ii, jj])

            for c in state[itr]['c']:
                print 'Sampling obs model (cluster:', c, ')'
                data_indx = numpy.where(numpy.array(state[itr]['c_for_each_y'].values()) == c)[0]
                mean_pdf = numpy.array([])
                for m in mean_vals:
                    logL = 0
                    for d in data_indx:
                        logL += self.get_gaussin_loglikelihood( data[d], {0:m[0],1:m[1]}, state[itr]['std'][c] )
                    mean_pdf = numpy.append(mean_pdf, logL)
                std_pdf = numpy.array([])
                for std in std_vals:
                    logL = 0
                    for d in data_indx:
                        logL += self.get_gaussin_loglikelihood( data[d], state[itr]['mu'][c], {0:std[0],1:std[1]} )
                    std_pdf = numpy.append(std_pdf, logL)

                ## SAMPLE NEW VALUES FOR OBS MODEL
                mean_sample_indx = self.draw_sample(mean_pdf)
                state[itr]['mu'][c][0] = mean_vals[mean_sample_indx][0];
                state[itr]['mu'][c][1] = mean_vals[mean_sample_indx][1];
            
                std_sample_indx = self.draw_sample(std_pdf)
                state[itr]['std'][c][0] = std_vals[std_sample_indx][0];
                state[itr]['std'][c][1] = std_vals[std_sample_indx][1];





DEBUG_VIS = 1
NUM_CLUSTERS = 3
mean = dict(); std=dict()
mean[0] = numpy.zeros(NUM_CLUSTERS);mean[1] = numpy.zeros(NUM_CLUSTERS)
std[0] = numpy.zeros(NUM_CLUSTERS);std[1] = numpy.zeros(NUM_CLUSTERS)
mean[0][0] = 3;mean[1][0] = -6; 
mean[0][1] = 2;mean[1][1] = 7; 
mean[0][2] = -8;mean[1][2] = 4; 
#mean[0][3] = -8;mean[1][3] = -6; 
#mean[0][4] = -3;mean[1][4] = 2; 
#mean[0][5] = 7;mean[1][5] = 2; 

STD_DEV = 1.0
std[0][0] = STD_DEV;std[1][0] = STD_DEV; 
std[0][1] = STD_DEV;std[1][1] = STD_DEV; 
std[0][2] = STD_DEV;std[1][2] = STD_DEV; 

obj = DPMixture()
#data = obj.get_data(dst=".",stdscale=STD_DEV,_mean_=mean, _std_=std)
obj.data = pickle.load(open("training_set_1.0.pkl","rb"))
obj.data = obj.data[0].values()

#print obj.sample_crp(100,10)

obj.prior()
obj.chain()

