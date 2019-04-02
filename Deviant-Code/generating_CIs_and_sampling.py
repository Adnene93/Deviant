from time import time
import cProfile
import pstats
import csv
import unicodedata
from random import randint,uniform,random,lognormvariate
from copy import deepcopy
from itertools import chain,product,ifilter
from filterer.filter import filter_pipeline_obj 
from math import trunc,sqrt
from sys import stdout
from collections import deque
from operator import iand,ior
#import numpy as np
from bisect import bisect_left
from math import log
from bisect import bisect
from functools import partial
from operator import itemgetter 
from cachetools import cached,LRUCache
import datetime
from outcomeDatasetsProcessor.outcomeDatasetsProcessor import process_outcome_dataset
from enumerator.enumerator_attribute_complex import enumerator_complex_cbo_init_new_config,enumerator_complex_from_dataset_new_config,pattern_subsume_pattern,respect_order_complex_not_after_closure,encode_sup,enumerator_generate_random_miserum
from enumerator.enumerator_attribute_hmt import all_parents_tag_exclusive
import gc
from util.matrixProcessing import transformMatricFromDictToList,adaptMatrices,getInnerMatrix,getCompleteMatrix
from numpy.random import choice,multinomial
from numpy import average,mean,std,linspace
from enumerator.enumerator_attribute_hmt import all_parents_tag_exclusive
from math import isnan
from sortedcontainers import SortedList
from scipy.stats import sem,t,shapiro,describe,histogram
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from bottleneck import partsort,argpartsort,median


def weighted_average(subset_values):
	sum_v=0.
	sum_w=0.
	for x in subset_values:
		sum_v+=x[0]
		sum_w+=x[1]
	return sum_v/sum_w


def compute_empiric_distribution(values,k,n,error=0.05):
	edf=[]
	
	for _ in xrange(n):
		selected_batch_for_now=(values[x] for x in choice(range(len(values)), k, replace=False).tolist())
		statistic=weighted_average(selected_batch_for_now)
		edf.append(statistic)
	_, min_max, mean, var, skew, kurt = describe(edf)
	std=sqrt(var)
	x=linspace(mean - 3*std, mean + 3*std, 50)
	dist=norm.pdf(x, mean, std)
	s_dist=float(sum(dist))
	dist=[t/s_dist for t in dist]

	edf=sorted(edf)
	CIPercentiles=(edf[int(len(edf)*(error/2.))],edf[int(len(edf)*(1-error/2.))])
	CINormal=norm.interval(1.-error, loc=mean, scale=std)
	return x,dist,edf,CIPercentiles,CINormal


def generate_random_distribution_of_weights_and_values(min_v=0.,max_v=1.,min_w=0.,max_w=1.,size=1000):
	W=[random()*(max_w-min_w)+min_w for _ in range(size)]
	V=[random()*(max_v-min_v)+min_v for _ in range(size)]
	VW=[V[i]*W[i] for i in range(len(V))]
	values=[(x,y) for (x,y) in zip(VW,W)]
	return values

def generate_high_amplitude_distribution(min_v=0.,max_v=1.,min_w=0.,max_w=1.,size=10000):
	#W=sorted([(lognormvariate(0.,1.1))*(max_w-min_w)+min_w for _ in xrange(size)],reverse=False)
	#W=sorted([max(W)-x for x in W],reverse=False)
	#W=sorted([(lognormvariate(0.,1.1))*(max_w-min_w)+min_w for _ in xrange(size)],reverse=False)
	W=[random()*((max_w-min_w))+min_w*(k**5) for k in range(size)]
	V=sorted([random()*(max_v-min_v)+min_v for _ in range(size)])
	#V=sorted([(lognormvariate(0.,1.1))*(max_v-min_v)+min_v for _ in xrange(size)],reverse=False)
	VW=[V[i]*W[i] for i in range(len(V))]
	values=[(x,y) for (x,y) in zip(VW,W)]
	# for _ in range(len(values)):
	# 	print values[_]

	# print values[980:-1]
	#raw_input('...')
	return values


if __name__ == '__main__':
	colors=['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499', '#117733']
	n=1000
	
	# c,l,largeur,e = histogram([(100-lognormvariate(0.,0.6))*100 for _ in xrange(1000)], 50)
	# c=[t/sum(c) for t in c]
	# xaxis_edf=[l+largeur * i for i in range(50)]
	# plt.bar(xaxis_edf,c,largeur, alpha=0.6,color='#332288')
	# plt.show()
	# raw_input('.......')
	fig, baseAx = plt.subplots(figsize=(30,18))
	baseAx.tick_params(axis='x', labelsize=25)
	baseAx.set_xlabel(r'$\theta$',fontsize=25)
	baseAx.tick_params(axis='y', labelsize=25)
	baseAx.set_ylabel('Probability',fontsize=25)
	nb_draws=5000
	min_v=10
	max_v=1000
	min_w=20
	max_w=1000
	size=5000
	# W=[randint(20,100) for _ in range(n)]
	# #V=[float('%.2f'%(random()*(200)+1)) for _ in range(n)]
	# V=[float('%.2f'%(random())) for _ in range(n)]
	# VW=[V[i]*W[i] for i in range(len(V))]
	# values=[(x,y) for (x,y) in zip(VW,W)]

	values=generate_high_amplitude_distribution(min_v,max_v,min_w,max_w,size)
	#2values=generate_random_distribution_of_weights_and_values(min_v,max_v,min_w,max_w,size)
	#print values
	color_index=0
	
	for k in [15,20,25,45,70,100,200,300,500]:
		a,b,edf,CIPercentiles,CINormal=compute_empiric_distribution(values,k,int(nb_draws))
		print CIPercentiles
		#print shapiro(edf)
		#raw_input('...')
		c,l,largeur,e = histogram(edf, 50)
		c=[t/sum(c) for t in c]
		xaxis_edf=[l+largeur * i for i in range(50)]
		#plt.plot(a,b,label='Line '+str(k),color=colors[color_index])
		plt.bar(xaxis_edf,c,largeur, label='Hist '+str(k) + ' '+ str([str(float("%.3f"%float(x))) for x in CIPercentiles]),alpha=0.6,color=colors[color_index])
		#plt.plot(a,b,label='Line '+str(k)+ ' ' +str(CINormal),color=colors[color_index])
		plt.legend(loc='upper right', fancybox=True, framealpha=0.85,fontsize=25)
		color_index+=1
	#print  [str(item.get_text())  for item in baseAx.get_xticklabels()]
	#old_labels=['%.2f'%float(str(item.get_text()))  for item in baseAx.get_xticklabels()][:]
	# labels = [('%.f'%((float(str(item))/exauhaustive_time_spent)*100))+'\%'  for item in old_labels]
	# labels = [r'$'+item+'_{'+ itemnew+'}$'  for item,itemnew in zip(old_labels,labels)]
	#baseAx.set_xticklabels(labels)


	fig.tight_layout()
	plt.savefig('tmp_1.pdf')
	# plt.legend()	
	# plt.show()