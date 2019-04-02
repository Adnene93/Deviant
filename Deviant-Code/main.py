import argparse
from time import time
import cProfile
import pstats
import csv
import unicodedata
from random import randint,uniform,random
from copy import deepcopy
from itertools import chain,product,ifilter,combinations
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
from numpy.random import choice,multinomial,shuffle,geometric
from numpy import average,mean,std,linspace,zeros
from enumerator.enumerator_attribute_hmt import all_parents_tag_exclusive
from math import isnan
from sortedcontainers import SortedList
from scipy.stats import sem,t,shapiro,describe#,histogram
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from bottleneck import partsort,argpartsort,median
from util.csvProcessing import writeCSV,writeCSVwithHeader,readCSVwithHeader
from util.epstein import random_epstein_exact,random_epstein_exact_weird_correction
from util.jsonProcessing import readJSON,readJSON_stringifyUnicodes
from util.distanceFunctions import distance_nominal,distance_categorical,distance_ordinal_simple,distance_ordinal_complex
import os
import shutil
from math import ceil




from util.approximations import expectation_ratio_approx,variance_ratio_approx,expectation_ratio_real,variance_ratio_real,CI_approx,CI_real,CI_approx_empirical_comparison,CI_approx_second,CI_approx_second_with_p_value,prediction_interval,minimum_k_valid_approx

def del_from_list_by_index(l,del_indexes):
	del_indexes_new_indexes=[];del_indexes_new_indexes_append=del_indexes_new_indexes.append
	if len(del_indexes):
		del l[del_indexes[0]]
		del_indexes_new_indexes_append(del_indexes[0])
		for k in range(1,len(del_indexes)):
			del_indexes_new_indexes_append((del_indexes[k]-del_indexes[k-1])+del_indexes_new_indexes[k-1]-1)
			del l[del_indexes_new_indexes[-1]]#l[del_indexes[k]-del_indexes[k-1]]


def find_subset_with_maximum_weighted_average_dynamicProgramming(Values,Weights,k):
	Stages=[]
	stage=[]
	for i in range(len(Values)):
		stage.append([i+1,Values[i],Weights[i]])
	Stages.append(stage)
	for i in range(1,k):
		stage=[]
		for (elem_pos,elem_value,elem_weight) in Stages[-1]:
			for j in range(elem_pos,len(Values)):
				stage.append((j+1,elem_value+(Weights[j]/elem_weight)*(Values[j]-elem_value),elem_weight+Weights[j]))
		Stages.append(stage)
		#print stage
	return max(stage[-1],key=itemgetter(1))[1]











def pre_compute_base_aggregates_for_kripendorff(all_entities_to_individuals_outcomes,domain_of_possible_outcomes,all_individuals_to_entities_outcomes,distance_function=distance_nominal,set_entities_to_consider=None,set_individuals_to_consider=None): #
	if set_individuals_to_consider is None:
		set_individuals_to_consider=set(all_individuals_to_entities_outcomes)
	if set_entities_to_consider is None:
		set_entities_to_consider=set(all_entities_to_individuals_outcomes)
	#set_entities_to_consider={e for e in set_entities_to_consider if len(all_entities_to_individuals_outcomes[e])>=2}	
	set_entities_to_consider=set(filter(lambda e:len(set(all_entities_to_individuals_outcomes[e])&set_individuals_to_consider)>=2,set_entities_to_consider))
	#RECONSIDER ONLY ITEMS THAT HAD BEEN VOTED AT LEAST BY TWO INDIVIDUALS TODO

	nb_raters=len({i for i in set_individuals_to_consider if len(all_individuals_to_entities_outcomes[i])>0}) #it is a sum of wieghts of the considered groups
	base_aggregates={}
	#marginal_infos={o:0 for o in domain_of_possible_outcomes}
	
	#########FLAT_COUTCOME#######
	
	flat_outcomes=[all_entities_to_individuals_outcomes[e][i] for e in set_entities_to_consider for i in set(all_entities_to_individuals_outcomes[e])&set_individuals_to_consider ]
	marginal_infos={o:float(flat_outcomes.count(o)) for o in domain_of_possible_outcomes}
	marginal_infos['nb_outcomes']=float(len(flat_outcomes))
	marginal_infos['nb_raters']=float(nb_raters) 
	#############################

	count_all_outcomes=0.
	for e in  set_entities_to_consider:
		entities_outcomes=all_entities_to_individuals_outcomes[e]
		entities_outcomes_to_consider=set(all_entities_to_individuals_outcomes[e])&set_individuals_to_consider
		entities_outcomes_counting={o:0 for o in domain_of_possible_outcomes}
		nb_outcomes_e=0.
		for i in entities_outcomes_to_consider:
			io=entities_outcomes[i] #if we have a distribution we need to make a pass (the ideal is to switch again to a distribution even if we have a sole individual)
			entities_outcomes_counting[io]+=1 #it is a sum of all ratings rather than 
			#marginal_infos[io]+=1
			nb_outcomes_e+=1 # same here
		entities_outcomes_counting['nb_outcomes']=nb_outcomes_e 
		
		
		e_dis_obs=0.
		probability_distribution_of_concording_peers_by_e=[]
		probability_distribution_of_concording_peers_by_e_candidates_names=[]
		for cindex in range(len(domain_of_possible_outcomes)):
			c=domain_of_possible_outcomes[cindex]
			for kindex in range(len(domain_of_possible_outcomes)):#range(cindex,len(domain_of_possible_outcomes)):
				k=domain_of_possible_outcomes[kindex]
				entities_outcomes_counting_c=entities_outcomes_counting[c]
				entities_outcomes_counting_k=entities_outcomes_counting[k] if k!=c else entities_outcomes_counting[k]-1
				proba_to_append=((entities_outcomes_counting_c)*(entities_outcomes_counting_k))/(nb_outcomes_e*(nb_outcomes_e-1)) if k!=c else ((entities_outcomes_counting_c)*(entities_outcomes_counting_k))/(nb_outcomes_e*(nb_outcomes_e-1))
				probability_distribution_of_concording_peers_by_e.append(proba_to_append)
				probability_distribution_of_concording_peers_by_e_candidates_names.append((c,k))
				e_dis_obs+=distance_function(c,k,domain_of_possible_outcomes=domain_of_possible_outcomes,marginal_infos=marginal_infos)*((entities_outcomes_counting[c]*entities_outcomes_counting[k])/(entities_outcomes_counting['nb_outcomes']*(entities_outcomes_counting['nb_outcomes']-1)))
			

		entities_outcomes_counting['probability_peers_distribution']=probability_distribution_of_concording_peers_by_e
		entities_outcomes_counting['candidates_names']=probability_distribution_of_concording_peers_by_e_candidates_names
		entities_outcomes_counting['quantity']=e_dis_obs
		# print e
		# for ix,x in enumerate(probability_distribution_of_concording_peers_by_e):
		# 	print '\t',probability_distribution_of_concording_peers_by_e_candidates_names[ix],':',probability_distribution_of_concording_peers_by_e[ix]
		# raw_input('....')
		base_aggregates[e]=entities_outcomes_counting
		count_all_outcomes+=nb_outcomes_e
		#print entities_outcomes_counting.keys()
		#raw_input('...')
	#marginal_infos['nb_outcomes']=count_all_outcomes
	#marginal_infos['nb_raters']=nb_raters


	disagreement_expected=0.
	for cindex in range(len(domain_of_possible_outcomes)):
			c=domain_of_possible_outcomes[cindex]
			for kindex in range(0,len(domain_of_possible_outcomes)):#range(0,len(domain_of_possible_outcomes)):
				k=domain_of_possible_outcomes[kindex]
				disagreement_expected+=distance_function(c,k,domain_of_possible_outcomes=domain_of_possible_outcomes,marginal_infos=marginal_infos)*((marginal_infos[c]*marginal_infos[k])/(marginal_infos['nb_outcomes']*(marginal_infos['nb_outcomes']-1)))
	
	#marginal_infos['disagreement_expected']=disagreement_expected
	marginal_infos['disagreement_expected']=disagreement_expected
	#print marginal_infos
	#print marginal_infos_2,'//',marginal_infos
	#raw_input('...')
	return base_aggregates,marginal_infos



#THIS IS NOT REALLY REALLY POSSIBLE SINCE I can't know what is exactly the number of raters that voted (intersecton between two entities is possible)
def pre_compute_base_aggregates_for_kripendorff_generalized_for_pre_aggregated_distributions(all_entities_to_individuals_outcomes,domain_of_possible_outcomes,all_individuals_to_entities_outcomes,distance_function=distance_nominal,set_entities_to_consider=None,set_individuals_to_consider=None): #
	AM_I_DEALING_WITH_PREAGGREGATED_DISTRIBUTIONS=False
	
	if set_individuals_to_consider is None:
		set_individuals_to_consider=set(all_individuals_to_entities_outcomes)
	if set_entities_to_consider is None:
		set_entities_to_consider=set(all_entities_to_individuals_outcomes)
	
	if AM_I_DEALING_WITH_PREAGGREGATED_DISTRIBUTIONS:
		set_entities_to_consider=set(filter(lambda e:sum([sum(all_entities_to_individuals_outcomes[e][i]) for i in set_individuals_to_consider])>=2,set_entities_to_consider))
	else:
		set_entities_to_consider=set(filter(lambda e:len(set(all_entities_to_individuals_outcomes[e])&set_individuals_to_consider)>=2,set_entities_to_consider))
	nb_raters=len({i for i in set_individuals_to_consider if len(all_individuals_to_entities_outcomes[i])>0}) #it is a sum of wieghts of the considered groups
	base_aggregates={}
	
	#########FLAT_COUTCOME#######
	
	flat_outcomes=[all_entities_to_individuals_outcomes[e][i] for e in set_entities_to_consider for i in set(all_entities_to_individuals_outcomes[e])&set_individuals_to_consider ]
	marginal_infos={o:float(flat_outcomes.count(o)) for o in domain_of_possible_outcomes}
	marginal_infos['nb_outcomes']=float(len(flat_outcomes))
	marginal_infos['nb_raters']=float(nb_raters) 
	#############################

	count_all_outcomes=0.
	for e in  set_entities_to_consider:
		entities_outcomes=all_entities_to_individuals_outcomes[e]
		entities_outcomes_to_consider=set(all_entities_to_individuals_outcomes[e])&set_individuals_to_consider
		entities_outcomes_counting={o:0 for o in domain_of_possible_outcomes}
		nb_outcomes_e=0.
		for i in entities_outcomes_to_consider:
			io=entities_outcomes[i] #if we have a distribution we need to make a pass (the ideal is to switch again to a distribution even if we have a sole individual)
			entities_outcomes_counting[io]+=1 #it is a sum of all ratings rather than 
			nb_outcomes_e+=1 # same here
		entities_outcomes_counting['nb_outcomes']=nb_outcomes_e 
		
		
		e_dis_obs=0.
		probability_distribution_of_concording_peers_by_e=[]
		probability_distribution_of_concording_peers_by_e_candidates_names=[]
		for cindex in range(len(domain_of_possible_outcomes)):
			c=domain_of_possible_outcomes[cindex]
			for kindex in range(len(domain_of_possible_outcomes)):#range(cindex,len(domain_of_possible_outcomes)):
				k=domain_of_possible_outcomes[kindex]
				entities_outcomes_counting_c=entities_outcomes_counting[c]
				entities_outcomes_counting_k=entities_outcomes_counting[k] if k!=c else entities_outcomes_counting[k]-1
				proba_to_append=((entities_outcomes_counting_c)*(entities_outcomes_counting_k))/(nb_outcomes_e*(nb_outcomes_e-1)) if k!=c else ((entities_outcomes_counting_c)*(entities_outcomes_counting_k))/(nb_outcomes_e*(nb_outcomes_e-1))
				probability_distribution_of_concording_peers_by_e.append(proba_to_append)
				probability_distribution_of_concording_peers_by_e_candidates_names.append((c,k))
				e_dis_obs+=distance_function(c,k,domain_of_possible_outcomes=domain_of_possible_outcomes,marginal_infos=marginal_infos)*((entities_outcomes_counting[c]*entities_outcomes_counting[k])/(entities_outcomes_counting['nb_outcomes']*(entities_outcomes_counting['nb_outcomes']-1)))
			

		entities_outcomes_counting['probability_peers_distribution']=probability_distribution_of_concording_peers_by_e
		entities_outcomes_counting['candidates_names']=probability_distribution_of_concording_peers_by_e_candidates_names
		entities_outcomes_counting['quantity']=e_dis_obs
		base_aggregates[e]=entities_outcomes_counting
		count_all_outcomes+=nb_outcomes_e


	disagreement_expected=0.
	for cindex in range(len(domain_of_possible_outcomes)):
			c=domain_of_possible_outcomes[cindex]
			for kindex in range(0,len(domain_of_possible_outcomes)):#range(0,len(domain_of_possible_outcomes)):
				k=domain_of_possible_outcomes[kindex]
				disagreement_expected+=distance_function(c,k,domain_of_possible_outcomes=domain_of_possible_outcomes,marginal_infos=marginal_infos)*((marginal_infos[c]*marginal_infos[k])/(marginal_infos['nb_outcomes']*(marginal_infos['nb_outcomes']-1)))
	marginal_infos['disagreement_expected']=disagreement_expected
	return base_aggregates,marginal_infos


def reliability(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal):
	disagreement_expected=0.
	disagreement_observed=0.
	# for cindex in range(len(domain_of_possible_outcomes)):
	# 	c=domain_of_possible_outcomes[cindex]
	# 	for kindex in range(len(domain_of_possible_outcomes)):#range(0,len(domain_of_possible_outcomes)):
	# 		k=domain_of_possible_outcomes[kindex]
	# 		disagreement_expected+=distance_function(c,k)*((marginal_infos[c]*marginal_infos[k])/(marginal_infos['nb_outcomes']*(marginal_infos['nb_outcomes']-1)))
	
	disagreement_expected=marginal_infos['disagreement_expected']
	number_of_pairable_values_context=0.

	for e in set_of_ballots:
		e_base_aggregates=base_aggregates[e]
		if False:
			e_dis_obs=0.
			for cindex in range(len(domain_of_possible_outcomes)):
				c=domain_of_possible_outcomes[cindex]

				for kindex in range(0,cindex)+range(0,len(domain_of_possible_outcomes)):
					k=domain_of_possible_outcomes[kindex]
					e_dis_obs+=distance_function(c,k,domain_of_possible_outcomes=domain_of_possible_outcomes,marginal_infos=marginal_infos)*((e_base_aggregates[c]*e_base_aggregates[k])/(e_base_aggregates['nb_outcomes']*(e_base_aggregates['nb_outcomes']-1)))
		else:
			e_dis_obs=e_base_aggregates['quantity']
		disagreement_observed+=e_base_aggregates['nb_outcomes']*e_dis_obs
		number_of_pairable_values_context+=e_base_aggregates['nb_outcomes']
	disagreement_observed=disagreement_observed/number_of_pairable_values_context

	#print 'GOOD RELIABILITY = ',disagreement_observed,disagreement_expected
	return 1-disagreement_observed/disagreement_expected



def reliability_as_kilem(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal):
	rbar=(1/float(len(set_of_ballots))) * sum(base_aggregates[e]['nb_outcomes'] for e in set_of_ballots)
	n = float(len(set_of_ballots))
	#r = marginal_infos['nb_outcomes']
	r = marginal_infos['nb_raters']
	#r = 
	dict_per_e={}
	# print domain_of_possible_outcomes,rbar,r
	# raw_input('....')


	p_a=0.
	p_e=0.
	# for k in marginal_infos:
	# 	print marginal_infos[k]
	for i in set_of_ballots:
		e_base_aggregates=base_aggregates[i]

		p_e_i=sum(((1/n) * marginal_infos[k] / r ) * (e_base_aggregates[k]/rbar) for k in domain_of_possible_outcomes) - ((e_base_aggregates['nb_outcomes'] - rbar)/rbar)
		p_a_i =sum((e_base_aggregates[k]*(e_base_aggregates[k]-1))/(rbar * (e_base_aggregates['nb_outcomes']-1)) for k in domain_of_possible_outcomes) 
		dict_per_e[i]={'p_e_i':p_e_i, 'p_a_i':p_a_i}
		p_a+=p_a_i
		p_e+=p_e_i
	p_a= (1/n)* p_a_i
	p_e= (1/n)* p_e_i
	alpha_k_hat=0.
	for i in set_of_ballots:
		p_a_i=dict_per_e[i]['p_a_i']
		r_i=e_base_aggregates['nb_outcomes']
		epsilon_n=(1/(marginal_infos['nb_outcomes']))#(1+1./(rbar*n-1))#(1/(marginal_infos['nb_outcomes']))
		p_a_epislon_n_per_i= (1-epsilon_n)*(p_a_i - p_a* ((r_i-rbar)/rbar))+(epsilon_n)
		dict_per_e[i]['p_a_epislon_n_per_i']=p_a_epislon_n_per_i
		alpha_k_per_i=(p_a_epislon_n_per_i - p_e)*(1-p_e)
		dict_per_e[i]['alpha_k_per_i']=alpha_k_per_i
		alpha_k_hat+=alpha_k_per_i
	alpha_k_hat=(1/n)*alpha_k_hat
	alpha_k=0.
	for i in set_of_ballots:

		alpha_k_star= dict_per_e[i]['alpha_k_per_i'] - ((1 - alpha_k_hat) * (dict_per_e[i]['p_e_i']-p_e))/(1-p_e)
		alpha_k+=alpha_k_star
	return alpha_k/n




def reliability_as_kilem_2(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal):
	


	rbar=(1/float(len(set_of_ballots))) * sum(base_aggregates[e]['nb_outcomes'] for e in set_of_ballots)
	n = float(len(set_of_ballots))
	#r = marginal_infos['nb_outcomes']
	r = marginal_infos['nb_raters']
	#r = 
	dict_per_e={}
	# print domain_of_possible_outcomes,rbar,r
	# raw_input('....')

	p_a=(1./n)*sum( sum((base_aggregates[i][k]*(base_aggregates[i][k]-1))/(rbar*(base_aggregates[i]['nb_outcomes']-1)) for k in domain_of_possible_outcomes)  for i in set_of_ballots)
	d_o = 1 - p_a
	
	#d_o = (1/(n*rbar))* sum(base_aggregates[i]['nb_outcomes']*(1-sum((base_aggregates[i][k]*(base_aggregates[i][k]-1))/(base_aggregates[i]['nb_outcomes']*(base_aggregates[i]['nb_outcomes']-1)) for k in domain_of_possible_outcomes)) for i in set_of_ballots )



	p_e=sum(((1./(n*r))*sum(base_aggregates[i][k] for i in set_of_ballots))**2 for k in domain_of_possible_outcomes)


	#d_e=(1.+1./(n*rbar-1))- ((n*(r**2))/(rbar*(n*rbar-1)))*p_e 
	d_e=1 - ((1/(n*rbar-1))*(((n*(r**2))/rbar)*p_e-1))   
	#print 'BAD RELIABILITY = ',d_o,d_e
	return 1-d_o/d_e




	# disagreement_expected=0.
	# disagreement_observed=0.
	# for cindex in range(len(domain_of_possible_outcomes)):
	# 	c=domain_of_possible_outcomes[cindex]
	# 	for kindex in range(0,len(domain_of_possible_outcomes)):
	# 		k=domain_of_possible_outcomes[kindex]
	# 		disagreement_expected+=distance_function(c,k)*((marginal_infos[c]*marginal_infos[k])/(marginal_infos['nb_outcomes']*(marginal_infos['nb_outcomes']-1)))
	# number_of_pairable_values_context=0.

	# for e in set_of_ballots:
	# 	e_base_aggregates=base_aggregates[e]
	# 	if False:
	# 		e_dis_obs=0.
	# 		for cindex in range(len(domain_of_possible_outcomes)):
	# 			c=domain_of_possible_outcomes[cindex]

	# 			for kindex in range(0,len(domain_of_possible_outcomes)):
	# 				k=domain_of_possible_outcomes[kindex]
	# 				e_dis_obs+=distance_function(c,k)*((e_base_aggregates[c]*e_base_aggregates[k])/(e_base_aggregates['nb_outcomes']*(e_base_aggregates['nb_outcomes']-1)))
	# 	else:
	# 		e_dis_obs=e_base_aggregates['quantity']
	# 	disagreement_observed+=e_base_aggregates['nb_outcomes']*e_dis_obs
	# 	number_of_pairable_values_context+=e_base_aggregates['nb_outcomes']
	# disagreement_observed=disagreement_observed/number_of_pairable_values_context


	# return 1-disagreement_observed/disagreement_expected


def get_VW_W(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal):
	disagreement_expected=marginal_infos['disagreement_expected']
	disagreement_observed=0.
	number_of_pairable_values_context=0.
	VW=[];VW_append=VW.append
	W=[];W_append=W.append
	for e in set_of_ballots:
		e_base_aggregates=base_aggregates[e]
		disagreement_observed+=e_base_aggregates['nb_outcomes']*e_base_aggregates['quantity']
		number_of_pairable_values_context+=e_base_aggregates['nb_outcomes']
		VW_append(e_base_aggregates['quantity']*e_base_aggregates['nb_outcomes'])
		W_append(e_base_aggregates['nb_outcomes'])
	
	disagreement_observed=disagreement_observed/number_of_pairable_values_context
	return VW,W,disagreement_expected
	
def get_vw_w_all_ready(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal):
	disagreement_expected=marginal_infos['disagreement_expected']
	disagreement_observed=0.
	number_of_pairable_values_context=0.
	VW=[];VW_append=VW.append
	W=[];W_append=W.append
	for e in set_of_ballots:
		e_base_aggregates=base_aggregates[e]
		disagreement_observed+=e_base_aggregates['nb_outcomes']*e_base_aggregates['quantity']
		number_of_pairable_values_context+=e_base_aggregates['nb_outcomes']
		VW_append(e_base_aggregates['quantity'] - (1./disagreement_expected) * (e_base_aggregates['quantity']*e_base_aggregates['nb_outcomes']))
		W_append(e_base_aggregates['nb_outcomes'])
	
	disagreement_observed=disagreement_observed/number_of_pairable_values_context
	return VW,W,disagreement_expected




def fast_reliability_with_bounds(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal,support_threshold=15,compute_bounds=False):
	
	TIGHT_BOUND=True

	disagreement_expected=0.
	disagreement_observed=0.
	# for cindex in range(len(domain_of_possible_outcomes)):
	# 	c=domain_of_possible_outcomes[cindex]
	# 	for kindex in range(0,len(domain_of_possible_outcomes)):#range(0,len(domain_of_possible_outcomes)):
	# 		k=domain_of_possible_outcomes[kindex]
	# 		disagreement_expected+=distance_function(c,k)*((marginal_infos[c]*marginal_infos[k])/(marginal_infos['nb_outcomes']*(marginal_infos['nb_outcomes']-1)))
	
	disagreement_expected=marginal_infos['disagreement_expected']
	number_of_pairable_values_context=0.

	VW=[];VW_append=VW.append
	W=[];W_append=W.append
	#print len(set_of_ballots)
	for e in set_of_ballots:
		e_base_aggregates=base_aggregates[e]
		disagreement_observed+=e_base_aggregates['nb_outcomes']*e_base_aggregates['quantity']
		number_of_pairable_values_context+=e_base_aggregates['nb_outcomes']
		if compute_bounds:
			VW_append(e_base_aggregates['quantity']*e_base_aggregates['nb_outcomes'])
			W_append(e_base_aggregates['nb_outcomes'])


	disagreement_observed=disagreement_observed/number_of_pairable_values_context
	
	alpha=1-disagreement_observed/disagreement_expected
	LBALPHA=-1*float('inf')
	UBALPHA=1.
	if compute_bounds:
		
		
		
		
		if TIGHT_BOUND:
			VALUES=[(x,y) for (x,y) in zip(VW,W)]
			FUNCTION_TO_USE=epstein_max_and_min #newton_optimized
			LB_TIGHT=FUNCTION_TO_USE(VALUES,support_threshold,bottom=True) #TIMOTHE
			UB_TIGHT=FUNCTION_TO_USE(VALUES,support_threshold)
			LBALPHA_TIGHT=1-UB_TIGHT/disagreement_expected
			UBALPHA_TIGHT=1-LB_TIGHT/disagreement_expected
			return LBALPHA_TIGHT,alpha,UBALPHA_TIGHT
		else:
			VW.sort()
			W.sort()
			LB=sum(VW[:support_threshold])/float(sum(W[-support_threshold:]))
			UB=sum(VW[-support_threshold:])/float(sum(W[:support_threshold]))
			
			LBALPHA=1-UB/disagreement_expected
			UBALPHA=1-LB/disagreement_expected
			return LBALPHA,alpha,UBALPHA
		# print LBALPHA,alpha,UBALPHA

		# #print LBALPHA,alpha,UBALPHA
		# raw_input('...')
		#raw_input('...')
		
	else:
		return alpha


def multinomiale(*args,**kwargs):
	return multinomial(*args,**kwargs)



def reliability_CI_KILEM_GWET_BLB(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,fullset_of_ballots,distance_function=distance_nominal,error=0.01,nb_per_bag=50,nb_bootstraps=5,one_tail_left=False,one_tail_right=False):
	print 'I AM HERE :)'
	CONFIDENCE_INTERVAL_TYPE='PERCENTILES'#'NORMAL' #'NORMAL' 'PERCENTILES'
	if one_tail_left:
		CONFIDENCE_INTERVAL_TYPE='ONE_TAIL_LEFT'
	elif one_tail_right:
		CONFIDENCE_INTERVAL_TYPE='ONE_TAIL_RIGHT'

	#CONFIDENCE_INTERVAL_TYPE='PERCENTILES' #'BACKWARDS' 'Normal'
	nb_iterations=0
	disagreement_expected=0.
	nb_per_bag=int(ceil(1./(error/4.)))
	# print nb_per_bag
	# raw_input('...')
	reliability_from_original_data=fast_reliability_with_bounds(marginal_infos,base_aggregates,fullset_of_ballots,domain_of_possible_outcomes,distance_function=distance_function)
	disagreement_expected=marginal_infos['disagreement_expected']
	nb_samples_unit=len(set_of_ballots)
	list_of_unities=sorted(fullset_of_ballots)
	estimating=[]
	for step_bootstrap in xrange(nb_bootstraps):
		print 'step_bootstrap : ',step_bootstrap
		step_in_a_bag_bootstrap=0
		selected_batch_for_now=choice(list_of_unities, int(nb_samples_unit**0.7), replace=False).tolist()
		one_blb=[]
		for constituted_sample_unit in multinomiale(int(nb_samples_unit), [1./len(selected_batch_for_now)]*len(selected_batch_for_now),nb_per_bag):
			step_in_a_bag_bootstrap+=1
			#print 'step_bootstrap', step_in_a_bag_bootstrap
			
			disagreement_observed_unit=0.
			number_of_pairable_values_context=0.
			for nb_ind,nb in enumerate(constituted_sample_unit):
				if nb==0:
					continue
				e_dis_obs=0.
				e_dis_obs_size=0.
				e_base_aggregates=base_aggregates[selected_batch_for_now[nb_ind]]
				nb_draw_by_unit=(e_base_aggregates['nb_outcomes']*(e_base_aggregates['nb_outcomes']-1))*nb
				probability_distribution=e_base_aggregates['probability_peers_distribution']
				list_of_candidates=e_base_aggregates['candidates_names']
				for i,v in enumerate(multinomiale(int(nb_draw_by_unit), probability_distribution)):
					c,k=list_of_candidates[i]
					nb_iterations+=1

					e_dis_obs+=(distance_function(c,k,domain_of_possible_outcomes=domain_of_possible_outcomes,marginal_infos=marginal_infos)*v)/nb_draw_by_unit #TODO
				number_of_pairable_values_context+=nb_draw_by_unit
				disagreement_observed_unit+=nb_draw_by_unit*e_dis_obs
			disagreement_observed_unit=disagreement_observed_unit/number_of_pairable_values_context
			one_blb.append(1-disagreement_observed_unit/disagreement_expected)
		

		one_blb.sort()
		if CONFIDENCE_INTERVAL_TYPE=='BACKWARDS':
			estimating.append([(2)*reliability_from_original_data-one_blb[int(len(one_blb)*(1-error/2.))],(2)*reliability_from_original_data-one_blb[int(len(one_blb)*(error/2.))]])
		if CONFIDENCE_INTERVAL_TYPE=='PERCENTILES':
			#print one_blb[-1],one_blb[-1]
			estimating.append([one_blb[int(len(one_blb)*(error/2.))],one_blb[int(len(one_blb)*(1-error/2.))]])
		elif CONFIDENCE_INTERVAL_TYPE == 'ONE_TAIL_LEFT':
			estimating.append([one_blb[int(len(one_blb)*(error))],float('+inf')])
		elif CONFIDENCE_INTERVAL_TYPE == 'ONE_TAIL_RIGHT':	
			estimating.append([float('-inf'),one_blb[int(len(one_blb)*(1-error))]])
		print estimating[-1]
		if False: #it follows normal distribution wilk-shapiro test
			_, min_max, mean, var, skew, kurt = describe(one_blb)
			#std=sqrt(var)
			#print norm.interval(1-error, loc=mean, scale=sem(one_blb)),t.interval(1-error, len(one_blb)-1, loc=mean, scale=sem(one_blb)),estimating[-1],var,mean #http://www.psychology.mcmaster.ca/bennett/boot09/confInt.pdf
			print t.interval(1-error, len(one_blb)-1, loc=mean, scale=var**(1/2.)),estimating[-1],reliability_from_original_data
	print 'CI =  ', sum(x[0] for x in estimating)/float(len(estimating)),sum(x[1] for x in estimating)/float(len(estimating)), 'actual Krippendorf : ',reliability_from_original_data
	return sum(x[0] for x in estimating)/float(len(estimating)),sum(x[1] for x in estimating)/float(len(estimating))








def reliability_CI_WITHOUT_REPLACEMENT(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,fullset_of_ballots,distance_function=distance_nominal,error=0.0001,nb_per_bag=100,nb_bootstraps=1,one_tail_left=False,one_tail_right=False):
	CONFIDENCE_INTERVAL_TYPE='NORMAL' #'NORMAL' 'PERCENTILES'
	if one_tail_left:
		CONFIDENCE_INTERVAL_TYPE='ONE_TAIL_LEFT'
	elif one_tail_right:
		CONFIDENCE_INTERVAL_TYPE='ONE_TAIL_RIGHT'


	nb_iterations=0
	disagreement_expected=0.
	reliability_from_original_data=fast_reliability_with_bounds(marginal_infos,base_aggregates,fullset_of_ballots,domain_of_possible_outcomes,distance_function=distance_function)
	disagreement_expected=marginal_infos['disagreement_expected']

	##disagreement_expected*=2.
	nb_samples_unit=len(set_of_ballots)
	list_of_unities=sorted(fullset_of_ballots)
	estimating=[]
	for _ in xrange(nb_bootstraps*nb_per_bag):
		selected_batch_for_now=choice(list_of_unities, int(nb_samples_unit), replace=False).tolist()
		disagreement_observed_unit=0.
		number_of_pairable_values_context=0.
		for u in selected_batch_for_now:
			e_base_aggregates=base_aggregates[u]
			e_dis_obs=0.
			e_dis_obs_size=0.
			nb_draw_by_unit=(e_base_aggregates['nb_outcomes']*(e_base_aggregates['nb_outcomes']-1))
			probability_distribution=e_base_aggregates['probability_peers_distribution']
			list_of_candidates=e_base_aggregates['candidates_names']
			for i,v in enumerate(multinomiale(int(nb_draw_by_unit), probability_distribution)):
				c,k=list_of_candidates[i]
				nb_iterations+=1
				e_dis_obs+=(distance_function(c,k)*v)/nb_draw_by_unit
			number_of_pairable_values_context+=nb_draw_by_unit
			disagreement_observed_unit+=nb_draw_by_unit*e_dis_obs
		disagreement_observed_unit=disagreement_observed_unit/number_of_pairable_values_context
		estimating.append(1-disagreement_observed_unit/disagreement_expected)
		#print estimating[-1]
	#print shapiro(estimating)
	#raw_input('....')
	CI=(-1.,1.)
	if CONFIDENCE_INTERVAL_TYPE == 'NORMAL':
		_, min_max, mean, var, skew, kurt = describe(estimating)
		std=sqrt(var)
		#CI=(mean-1.96*std,mean+1.96*std)
		CI=norm.interval(1.-error, loc=mean, scale=std)
		#print std
		#x=fast_reliability_with_bounds(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal)
		#print x,CI,norm.cdf(x, loc=mean, scale=std)
		#print CI,norm.interval(0.999, loc=mean, scale=std)
		#raw_input('.....')

	elif CONFIDENCE_INTERVAL_TYPE == 'PERCENTILES':
		estimating.sort()
		CI=(estimating[int(len(estimating)*(error/2.))],estimating[int(len(estimating)*(1-error/2.))])
	elif CONFIDENCE_INTERVAL_TYPE == 'ONE_TAIL_LEFT':
		estimating.sort()
		CI=(estimating[int(len(estimating)*(error))],+float('inf'))
	elif CONFIDENCE_INTERVAL_TYPE == 'ONE_TAIL_RIGHT':	
		estimating.sort()
		CI=(float('-inf'),estimating[int(len(estimating)*(1-error))])
		# one_blb.sort()
		# if CONFIDENCE_INTERVAL_TYPE=='BACKWARDS':
		# 	estimating.append([(2)*reliability_from_original_data-one_blb[int(len(one_blb)*(1-error/2.))],(2)*reliability_from_original_data-one_blb[int(len(one_blb)*(error/2.))]])
		# if CONFIDENCE_INTERVAL_TYPE=='PERCENTILES':
		# 	estimating.append([one_blb[int(len(one_blb)*(error/2.))],one_blb[int(len(one_blb)*(1-error/2.))]])
		# if False: #it follows normal distribution wilk-shapiro test
		# 	_, min_max, mean, var, skew, kurt = describe(one_blb)
		# 	#std=sqrt(var)
		# 	print norm.interval(1-error, loc=mean, scale=sem(one_blb)),t.interval(1-error, len(one_blb)-1, loc=mean, scale=sem(one_blb)),estimating[-1],var,mean #http://www.psychology.mcmaster.ca/bennett/boot09/confInt.pdf
	return CI




def get_sample(arr, n_iter=None, sample_size=10, fast=True):
	n=len(arr)
	if fast:
		start_idx = (n_iter * sample_size) % n
		if start_idx + sample_size >= n:
			shuffle(arr)

		return arr[start_idx:start_idx+sample_size] 
	else:
		return choice(arr, sample_size, replace=False)


def fast_reservoir_sampling(arr, sample_size):
	sample = []
	sample_append=sample.append
	# if callable is None:
	#     callable = lambda x: x

	j = sample_size
	for n, line in enumerate(arr):
		if n < sample_size:
			sample_append(line)
		else:
			if n < j:
				continue
			p = sample_size / n
			g = geometric(p)
			j = j + g
			replace = randint(0, sample_size-1)
			sample[replace] = line

	return sample


def collect_samples(arr,sample_size,n_samples,fast=False):
	shuffle(arr)
	for sample_n in xrange(n_samples):
		yield  get_sample(arr, n_iter=sample_n,sample_size=sample_size,fast=fast)
		#yield  fast_reservoir_sampling(arr, float(sample_size))
				




def reliability_CI_WITHOUT_REPLACEMENT_ON_SETS(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,fullset_of_ballots,distance_function=distance_nominal,error=0.01,nb_per_bag=5000,nb_bootstraps=1,one_tail_left=False,one_tail_right=False):
	CONFIDENCE_INTERVAL_TYPE='STUDENT'
	#CONFIDENCE_INTERVAL_TYPE='PERCENTILES'#'NORMAL' #'NORMAL' 'PERCENTILES'
	if one_tail_left:
		CONFIDENCE_INTERVAL_TYPE='ONE_TAIL_LEFT'
	elif one_tail_right:
		CONFIDENCE_INTERVAL_TYPE='ONE_TAIL_RIGHT'


	nb_iterations=0
	disagreement_expected=0.
	#reliability_from_original_data=fast_reliability_with_bounds(marginal_infos,base_aggregates,fullset_of_ballots,domain_of_possible_outcomes,distance_function=distance_function)
	disagreement_expected=marginal_infos['disagreement_expected']
	nb_samples_unit=len(set_of_ballots)
	list_of_unities=fullset_of_ballots[:]#Don't Compute fucking each time
	estimating=[]
	
	for selected_batch_for_now in collect_samples(list_of_unities,int(nb_samples_unit),nb_bootstraps*nb_per_bag,fast=False):
		estimating.append(fast_reliability_with_bounds(marginal_infos,base_aggregates,selected_batch_for_now,domain_of_possible_outcomes,distance_function=distance_function))
	# for _ in xrange(nb_bootstraps*nb_per_bag):
	# 	selected_batch_for_now=choice(list_of_unities, int(nb_samples_unit), replace=False).tolist()
	# 	estimating.append(fast_reliability_with_bounds(marginal_infos,base_aggregates,selected_batch_for_now,domain_of_possible_outcomes,distance_function=distance_function))

	CI=(-1.,1.)
	if CONFIDENCE_INTERVAL_TYPE == 'NORMAL':
		_, min_max, mean, var, skew, kurt = describe(estimating)
		std=sqrt(var)
		CI=norm.interval(1.-error, loc=mean, scale=std)
	elif CONFIDENCE_INTERVAL_TYPE == 'STUDENT':
		# prediction_interval()
		# _, min_max, mean, var, skew, kurt = describe(estimating)
		# std=sqrt(var)
		CI=prediction_interval(estimating,error)#t.interval(1.-error, loc=mean, scale=std)


	elif CONFIDENCE_INTERVAL_TYPE == 'PERCENTILES':
		estimating.sort()
		CI=(estimating[int(len(estimating)*(error/2.))],estimating[int(len(estimating)*(1-error/2.))])
	elif CONFIDENCE_INTERVAL_TYPE == 'ONE_TAIL_LEFT':
		estimating.sort()
		CI=(estimating[int(len(estimating)*(error))],+float('inf'))
	elif CONFIDENCE_INTERVAL_TYPE == 'ONE_TAIL_RIGHT':	
		estimating.sort()
		CI=(float('-inf'),estimating[int(len(estimating)*(1-error))])
	return CI




def reliability_CI_WITHOUT_REPLACEMENT_DISTRIBUTION(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,fullset_of_ballots,distance_function=distance_nominal,error=0.01,nb_per_bag=100,nb_bootstraps=5):
	CONFIDENCE_INTERVAL_TYPE='NORMAL' #'BACKWARDS' 'Normal'
	nb_iterations=0
	disagreement_expected=0.
	reliability_from_original_data=fast_reliability_with_bounds(marginal_infos,base_aggregates,fullset_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal)
	# for cindex in range(len(domain_of_possible_outcomes)):
	# 	c=domain_of_possible_outcomes[cindex]
	# 	for kindex in range(0,len(domain_of_possible_outcomes)):
	# 		k=domain_of_possible_outcomes[kindex]
	# 		disagreement_expected+=distance_function(c,k)*((marginal_infos[c]*marginal_infos[k])/(marginal_infos['nb_outcomes']*(marginal_infos['nb_outcomes']-1)))
	disagreement_expected=marginal_infos['disagreement_expected']
	#disagreement_expected*=2.
	nb_samples_unit=len(set_of_ballots)
	list_of_unities=sorted(fullset_of_ballots)
	estimating=[]
	for _ in xrange(nb_bootstraps*nb_per_bag):
		selected_batch_for_now=choice(list_of_unities, int(nb_samples_unit), replace=True).tolist()
		disagreement_observed_unit=0.
		number_of_pairable_values_context=0.
		for u in selected_batch_for_now:
			e_base_aggregates=base_aggregates[u]
			e_dis_obs=0.
			e_dis_obs_size=0.
			nb_draw_by_unit=(e_base_aggregates['nb_outcomes']*(e_base_aggregates['nb_outcomes']-1))
			probability_distribution=e_base_aggregates['probability_peers_distribution']
			list_of_candidates=e_base_aggregates['candidates_names']
			for i,v in enumerate(multinomiale(int(nb_draw_by_unit), probability_distribution)):
				c,k=list_of_candidates[i]
				nb_iterations+=1
				e_dis_obs+=(distance_function(c,k)*v)/nb_draw_by_unit
			number_of_pairable_values_context+=nb_draw_by_unit
			disagreement_observed_unit+=nb_draw_by_unit*e_dis_obs
		disagreement_observed_unit=disagreement_observed_unit/number_of_pairable_values_context
		estimating.append(1-disagreement_observed_unit/disagreement_expected)
		#print estimating[-1]
	
	CI=(-1.,1.)
	if CONFIDENCE_INTERVAL_TYPE == 'NORMAL':
		_, min_max, mean, var, skew, kurt = describe(estimating)
		std=sqrt(var)
		#CI=(mean-1.96*std,mean+1.96*std)
		CI=norm.interval(1.-error, loc=mean, scale=std)
	elif CONFIDENCE_INTERVAL_TYPE == 'PERCENTILES':
		estimating.sort()
		CI=(estimating[int(len(estimating)*(error/2.))],estimating[int(len(estimating)*(1-error/2.))])
	
	x=linspace(mean - 3*std, mean + 3*std, 50)
	dist=norm.pdf(x, mean, std)
	#dist=[dist[0]]+[dist[i]-dist[i-1] for i in range(1,len(dist))]
	s_dist=float(sum(dist))
	
	#print []
	#dist=[t/sum(dist) for t in dist]
	#print dist
	dist=[t/s_dist for t in dist]
	#print sum(dist)
	return x,dist,sorted(estimating)#mlab.normpdf(x, mean, std)

def reliability_CI_KILEM_GWET_BLB_FLEXIBLE(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,fullset_of_ballots,distance_function=distance_nominal,error=0.05,nb_per_bag=1000,nb_bootstraps=5):
	nb_iterations=0
	disagreement_expected=0.
	CINOW=(0,0)
	


	# reliability_from_original_data=fast_reliability_with_bounds(marginal_infos,base_aggregates,fullset_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal)
	# for cindex in range(len(domain_of_possible_outcomes)):
	# 	c=domain_of_possible_outcomes[cindex]
	# 	for kindex in range(0,len(domain_of_possible_outcomes)):
	# 		k=domain_of_possible_outcomes[kindex]
	# 		disagreement_expected+=distance_function(c,k)*((marginal_infos[c]*marginal_infos[k])/(marginal_infos['nb_outcomes']*(marginal_infos['nb_outcomes']-1)))
	reliability_from_original_data=fast_reliability_with_bounds(marginal_infos,base_aggregates,fullset_of_ballots,domain_of_possible_outcomes,distance_function=distance_function)
	disagreement_expected=marginal_infos['disagreement_expected']
	#disagreement_expected*=2.
	nb_samples_unit=len(set_of_ballots)
	list_of_unities=sorted(fullset_of_ballots)
	estimating=[]
	for _ in xrange(nb_bootstraps):
		selected_batch_for_now=choice(list_of_unities, int(nb_samples_unit**0.8), replace=False).tolist()
		one_blb=SortedList()
		convergence_rate=1.
		CIS=[]
		for constituted_sample_unit in multinomiale(int(nb_samples_unit), [1./len(selected_batch_for_now)]*len(selected_batch_for_now),nb_per_bag):
			disagreement_observed_unit=0.
			number_of_pairable_values_context=0.
			for nb_ind,nb in enumerate(constituted_sample_unit):
				if nb==0:
					continue
				e_dis_obs=0.
				e_dis_obs_size=0.
				e_base_aggregates=base_aggregates[selected_batch_for_now[nb_ind]]
				nb_draw_by_unit=(e_base_aggregates['nb_outcomes']*(e_base_aggregates['nb_outcomes']-1))*nb
				probability_distribution=e_base_aggregates['probability_peers_distribution']
				list_of_candidates=e_base_aggregates['candidates_names']
				for i,v in enumerate(multinomiale(int(nb_draw_by_unit), probability_distribution)):
					c,k=list_of_candidates[i]
					nb_iterations+=1
					e_dis_obs+=(distance_function(c,k,domain_of_possible_outcomes=domain_of_possible_outcomes,marginal_infos=marginal_infos)*v)/nb_draw_by_unit
				number_of_pairable_values_context+=nb_draw_by_unit
				disagreement_observed_unit+=nb_draw_by_unit*e_dis_obs
			disagreement_observed_unit=disagreement_observed_unit/number_of_pairable_values_context
			#one_blb.append()
			
			
			one_blb.add(1-disagreement_observed_unit/disagreement_expected)
			
			#CINOW=((2)*reliability_from_original_data-one_blb[int(len(one_blb)*(1-error/2.))],(2)*reliability_from_original_data-one_blb[int(len(one_blb)*(error/2.))])
			CINOW=[one_blb[int(len(one_blb)*(error/2.))],one_blb[int(len(one_blb)*(1-error/2.))]]
			#CINOW=((2)*reliability_from_original_data-one_blb[int(len(one_blb)*(1-error/2.))],(2)*reliability_from_original_data-one_blb[int(len(one_blb)*(error/2.))])
			#print CINOW
			CIS.append(CINOW)
			#print CIS

			if len(CIS)>=4:
				convergence_rate=max(map(lambda x : max(abs(CIS[-1][0]-x[0]),abs(CIS[-1][1]-x[1])),(CIS[y] for y in range(len(CIS)-4,len(CIS)-1)))) 
			#print CINOW
			
			if convergence_rate<0.01 and len(CIS)>=15:
				break

		#one_blb.sort()

		#estimating.append([(2)*reliability_from_original_data-one_blb[int(len(one_blb)*(1-error/2.))],(2)*reliability_from_original_data-one_blb[int(len(one_blb)*(error/2.))]])
		
		estimating.append([one_blb[int(len(one_blb)*(error/2.))],one_blb[int(len(one_blb)*(1-error/2.))]])
		print estimating[-1]
	return sum(x[0] for x in estimating)/float(len(estimating)),sum(x[1] for x in estimating)/float(len(estimating))

def reliability_CI_KILEM_GWET_BLB_BASIC(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,fullset_of_ballots,distance_function=distance_nominal):
	nb_iterations=0
	error=0.001
	disagreement_expected=0.
	#disagreement_observed=0.
	NB_BOOTSTRAPS=50
	
	reliability_from_original_data=reliability(marginal_infos,base_aggregates,fullset_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal)

	# for c in domain_of_possible_outcomes:
	# 	for k in domain_of_possible_outcomes:
	# 		disagreement_expected+=distance_function(c,k)*((marginal_infos[c]*marginal_infos[k])/(marginal_infos['nb_outcomes']*(marginal_infos['nb_outcomes']-1)))
	disagreement_expected=marginal_infos['disagreement_expected']

	nb_samples_unit=len(set_of_ballots)
	list_of_unities=sorted(fullset_of_ballots)
	estimating=[]
	for _ in xrange(NB_BOOTSTRAPS):
		selected_batch_for_now=choice(list_of_unities, int(nb_samples_unit**0.5), replace=False).tolist()

		one_blb=[]
		for constituted_sample_unit in multinomiale(int(nb_samples_unit), [1./len(selected_batch_for_now)]*len(selected_batch_for_now),50):
			#print constituted_sample_unit
			disagreement_observed_unit=0.
			number_of_pairable_values_context=0.
			for nb_ind,nb in enumerate(constituted_sample_unit):
				if nb==0:
					continue
				e_dis_obs=0.
				e_dis_obs_size=0.
				e_base_aggregates=base_aggregates[selected_batch_for_now[nb_ind]]
				nb_draw_by_unit=(e_base_aggregates['nb_outcomes']*(e_base_aggregates['nb_outcomes']-1))*nb
				probability_distribution=e_base_aggregates['probability_peers_distribution']
				list_of_candidates=e_base_aggregates['candidates_names']
				#list_of_generated_peers=[(list_of_candidates[i],v) for i,v in enumerate(multinomiale(int(nb_draw_by_unit), probability_distribution))]
				
				for i,v in enumerate(multinomiale(int(nb_draw_by_unit), probability_distribution)):
					c,k=list_of_candidates[i]
					nb_iterations+=1
					# if isnan(nb_draw_by_unit) or nb_draw_by_unit==0:
					# 	print nb_draw_by_unit,list_of_generated_peers,constituted_sample_unit
					e_dis_obs+=(distance_function(c,k)*v)/nb_draw_by_unit
					
					

				
				number_of_pairable_values_context+=nb_draw_by_unit
				disagreement_observed_unit+=nb_draw_by_unit*e_dis_obs
			disagreement_observed_unit=disagreement_observed_unit/number_of_pairable_values_context
			#print 1-disagreement_observed_unit/disagreement_expected
			one_blb.append(1-disagreement_observed_unit/disagreement_expected)
			
		one_blb.sort()


		#print sem(one_blb)

		#estimating.append([(2)*(sum(one_blb)/float(len(one_blb)))-one_blb[int(len(one_blb)*(1-error/2.))],(2)*(sum(one_blb)/float(len(one_blb)))-one_blb[int(len(one_blb)*(error/2.))]])

		#print returned_sorted[int(len(returned_sorted)*0.025)],returned_sorted[int(len(returned_sorted)*0.975)]
		
		estimating.append([one_blb[int(len(one_blb)*((error)/2.))],one_blb[int(len(one_blb)*((1-error)/2.))]])
		print estimating[-1]
		#raw_input('...............')
	print 'NB ITERATIONS ! '
	print nb_iterations
	return sum(x[0] for x in estimating)/float(len(estimating)),sum(x[1] for x in estimating)/float(len(estimating))


def reliability_CI_KILEM_GWET_BLB_STUDENTIZE(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,fullset_of_ballots,distance_function=distance_nominal):
	nb_iterations=0
	error=0.001
	disagreement_expected=0.
	#disagreement_observed=0.
	NB_BOOTSTRAPS=50
	
	reliability_from_original_data=reliability(marginal_infos,base_aggregates,fullset_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal)

	# for c in domain_of_possible_outcomes:
	# 	for k in domain_of_possible_outcomes:
	# 		disagreement_expected+=distance_function(c,k)*((marginal_infos[c]*marginal_infos[k])/(marginal_infos['nb_outcomes']*(marginal_infos['nb_outcomes']-1)))

	disagreement_expected=marginal_infos['disagreement_expected']

	nb_samples_unit=len(set_of_ballots)
	list_of_unities=sorted(fullset_of_ballots)
	estimating=[]
	for _ in xrange(NB_BOOTSTRAPS):
		selected_batch_for_now=choice(list_of_unities, int(nb_samples_unit**0.5), replace=False).tolist()

		one_blb=[]
		for constituted_sample_unit in multinomiale(int(nb_samples_unit), [1./len(selected_batch_for_now)]*len(selected_batch_for_now),50):
			#print constituted_sample_unit
			disagreement_observed_unit=0.
			number_of_pairable_values_context=0.
			for nb_ind,nb in enumerate(constituted_sample_unit):
				if nb==0:
					continue
				e_dis_obs=0.
				e_dis_obs_size=0.
				e_base_aggregates=base_aggregates[selected_batch_for_now[nb_ind]]
				nb_draw_by_unit=(e_base_aggregates['nb_outcomes']*(e_base_aggregates['nb_outcomes']-1))*nb
				probability_distribution=e_base_aggregates['probability_peers_distribution']
				list_of_candidates=e_base_aggregates['candidates_names']
				#list_of_generated_peers=[(list_of_candidates[i],v) for i,v in enumerate(multinomiale(int(nb_draw_by_unit), probability_distribution))]
				
				for i,v in enumerate(multinomiale(int(nb_draw_by_unit), probability_distribution)):
					c,k=list_of_candidates[i]
					nb_iterations+=1
					# if isnan(nb_draw_by_unit) or nb_draw_by_unit==0:
					# 	print nb_draw_by_unit,list_of_generated_peers,constituted_sample_unit
					e_dis_obs+=(distance_function(c,k)*v)/nb_draw_by_unit
					
					

				
				number_of_pairable_values_context+=nb_draw_by_unit
				disagreement_observed_unit+=nb_draw_by_unit*e_dis_obs
			disagreement_observed_unit=disagreement_observed_unit/number_of_pairable_values_context
			#print 1-disagreement_observed_unit/disagreement_expected
			one_blb.append(1-disagreement_observed_unit/disagreement_expected)
			
		one_blb.sort()


		standared_error=sem(one_blb)
		students=t.interval(1-error,50-1)
		print standared_error,reliability_from_original_data,students
		raw_input('...')
		
		estimating.append([reliability_from_original_data-students[1]*(standared_error),reliability_from_original_data+students[0]*(standared_error)])
		#estimating.append([(2)*(sum(one_blb)/float(len(one_blb)))-one_blb[int(len(one_blb)*(1-error/2.))],(2)*(sum(one_blb)/float(len(one_blb)))-one_blb[int(len(one_blb)*(error/2.))]])

		#print returned_sorted[int(len(returned_sorted)*0.025)],returned_sorted[int(len(returned_sorted)*0.975)]
		

		print estimating[-1]
		#raw_input('...............')
	print 'NB ITERATIONS ! '
	print nb_iterations
	return sum(x[0] for x in estimating)/float(len(estimating)),sum(x[1] for x in estimating)/float(len(estimating))







def printer_hmt(arr_tag_with_labels):
	ret={x[:x.find(' ')]:x for x in arr_tag_with_labels}
	tags=ret.viewkeys()
	tags=sorted(tags-reduce(set.union,[all_parents_tag_exclusive(x) for x in tags]))
	
	return [ret[x] for x in tags]


def pattern_printer(pattern,types_attributes):
	
	s=''
	for k in range(len(pattern)):
		if k<len(types_attributes):
			if types_attributes[k]=='simple':
				if len(pattern[k])>1:
					if True:
						s+= '*'+' '
					else:
						s+= str(pattern[k])+' '
				else:
					s+= str(pattern[k][0])+' '
			elif types_attributes[k] in {'themes','hmt'}:
				
				s+= str(printer_hmt(pattern[k]))+' '#str(pattern[k])+' '
			else:
				s+= str(pattern[k])+' '
		else:
			s+= str(pattern[k])+' '
		
	return s



def find_exeptional_contexts(considered_items_sorted,contexts_attributes,entities_id_attribute,marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal,support_threshold=10,quality_threshold=0.4):
	PRUNE=True
	NB_PATTERNS_VISITED=0.
	lists_of_patterns=[]

	enumerator_contexts=enumerator_complex_cbo_init_new_config(considered_items_sorted, contexts_attributes,threshold=support_threshold)
	reliability_full=fast_reliability_with_bounds(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal)
	for e_p,e_label,e_config in enumerator_contexts:
		NB_PATTERNS_VISITED+=1
		context_pat=pattern_printer(e_label,['themes'])

		entities_context=set(x[entities_id_attribute] for x in e_config['support'])
	
		LB,reliability_pattern,UB=fast_reliability_with_bounds(marginal_infos,base_aggregates,entities_context,domain_of_possible_outcomes,distance_function=distance_nominal,support_threshold=support_threshold,compute_bounds=True)
		quality_pattern=((reliability_pattern-reliability_full)**2)**(0.5)
		UBQUALITY=((UB-reliability_full)**2)**(0.5)
		LBQUALITY=((reliability_full-LB)**2)**(0.5)
		UBB=max(LBQUALITY,UBQUALITY)
		if PRUNE and UBB<quality_threshold:
			e_config['flag']=False
			continue
		if quality_pattern>=quality_threshold:
			lists_of_patterns.append((context_pat,reliability_pattern,len(entities_context),quality_pattern))
		
	lists_of_patterns=sorted(lists_of_patterns,key=lambda x:x[3],reverse=True)
	print 'NB_PATTERNS_VISITED : ', NB_PATTERNS_VISITED
	return lists_of_patterns




def find_exceptional_common_contexts_and_individuals(considered_individuals_sorted,
													 considered_items_sorted,
													 individuals_attributes,
													 contexts_attributes,
													 individuals_id_attribute,
													 entities_id_attribute,
													 all_entities_to_individuals_outcomes,
													 all_individuals_to_entities_outcomes,
													 domain_of_possible_outcomes,
													 distance_function=distance_nominal,
													 support_context_threshold=10,
													 support_individuals_threshold=10,
													 quality_threshold=0.1, #become the error in case of confidence_interval_quality=True
													 quality_measure='CONFLICTUAL',
													 confidence_interval_quality=False,
													 closure_operator=True,
													 pruning=True,
													 using_approxmations=False,
													 nb_random_sample=5000,
													 compute_bootstrap_ci=False,
													 compute_p_value=False,
													 concept_of_generality=True,
													 replace_ids=True,
													 VERBOSE=True):
	
	
	


	COMPUTE_BOOTSTRAPING_CI=compute_bootstrap_ci
	nb_explored=0
	nb_patterns=0
	STATS={

		'nb_entities':len(considered_items_sorted),
		'nb_individuals':len(considered_individuals_sorted),
		'nb_individuals_attributes':len(individuals_attributes),
		'nb_individuals_attributes_items':'TBA',
		'nb_contexts_attributes':len(contexts_attributes),
		'nb_contexts_attributes_items':'TBA',
		'support_individuals_threshold':support_individuals_threshold,
		'support_context_threshold':support_context_threshold,
		'quality_threshold':quality_threshold,
		'closure_operator':closure_operator,
		'pruning':pruning,
		'using_approxmations':using_approxmations,
		'nb_random_sample':nb_random_sample,

		'distance_function':'TBA',
		'timespent_execution':'TBA',
		'timespent_init':'TBA',
		'timespent_total':'TBA',
		'nb_explored':nb_explored,
		'nb_patterns':nb_patterns,


	}

	if support_individuals_threshold<1:
		support_individuals_threshold=int(len(considered_individuals_sorted)*support_individuals_threshold)
		print 'support_individuals_threshold=',support_individuals_threshold
	else:
		support_individuals_threshold=int(support_individuals_threshold)
	
	if support_context_threshold<1:
		support_context_threshold=int(len(considered_items_sorted)*support_context_threshold)
		print 'support_context_threshold=',support_context_threshold
	else:
		support_context_threshold=int(support_context_threshold)

	timespent=time()
	PRUNE=pruning
	ids_replaced=replace_ids
	lists_of_patterns=[]
	all_observed_individuals=set(all_individuals_to_entities_outcomes.keys())
	types_individuals_attributes=[x['type'] for x in individuals_attributes]


	indices_to_id_individuals=[u[individuals_id_attribute] for i,u in enumerate(considered_individuals_sorted)]
	id_individuals_to_indices={u:i for i,u in enumerate(indices_to_id_individuals)}
	indices_to_id_entities=[e[entities_id_attribute] for i,e in enumerate(considered_items_sorted)]
	id_entities_to_indices={e:i for i,e in enumerate(indices_to_id_entities)}

	all_observed_individuals_indices=set(id_individuals_to_indices[x] for x in all_observed_individuals) if not ids_replaced else all_observed_individuals
	if ids_replaced:
		print indices_to_id_entities==range(len(considered_items_sorted))
		print indices_to_id_individuals==range(len(considered_individuals_sorted))
		#raw_input('....')

	types_individuals_attributes=[x['type'] for x in  individuals_attributes]
	types_context_attributes=[x['type'] for x in  contexts_attributes]
	initConfig={'config':None,'attributes':None}

	##########
	if VERBOSE:
		nb_possible_individuals=0
		enum_test=enumerator_complex_cbo_init_new_config(considered_individuals_sorted, individuals_attributes,threshold=support_individuals_threshold,closed=closure_operator,config_init={'indices':all_observed_individuals_indices,'indices_bitset':encode_sup(sorted(all_observed_individuals_indices),len(considered_individuals_sorted))})
		for _ in enum_test:
			nb_possible_individuals+=1
		print 'nb_possible_group : ',nb_possible_individuals
	#########

	####################TEMPORARY############
	base_aggregates_all_users,marginal_infos_all_users=pre_compute_base_aggregates_for_kripendorff(all_entities_to_individuals_outcomes,domain_of_possible_outcomes,all_individuals_to_entities_outcomes,distance_function=distance_function)
	reliability_default=fast_reliability_with_bounds(marginal_infos_all_users,base_aggregates_all_users,set(all_entities_to_individuals_outcomes.keys()),domain_of_possible_outcomes,distance_function=distance_function)
	#########################################
	count_i_p=0
	enumerator_individuals=enumerator_complex_cbo_init_new_config(considered_individuals_sorted, individuals_attributes,threshold=support_individuals_threshold,closed=closure_operator,config_init={'indices':all_observed_individuals_indices,'indices_bitset':encode_sup(sorted(all_observed_individuals_indices),len(considered_individuals_sorted))})
	for i_p,i_label,i_config in enumerator_individuals:
		if VERBOSE:
			count_i_p+=1
			stdout.write('%s\r' % ('Percentage Done : ' + ('%.2f'%((count_i_p/float(nb_possible_individuals))*100))+ '%'));stdout.flush();
			

		group_pat=i_label
		
		individuals_support_indices=i_config['indices']
		individuals_support=set(indices_to_id_individuals[x] for x in i_config['indices']) if not ids_replaced else individuals_support_indices
		individuals_support_bitset=i_config['indices_bitset']
		# print sorted(individuals_support)
		# print sorted(all_observed_individuals)
		# raw_input('...')

		base_aggregates,marginal_infos=pre_compute_base_aggregates_for_kripendorff(all_entities_to_individuals_outcomes,domain_of_possible_outcomes,all_individuals_to_entities_outcomes,distance_function=distance_function,set_individuals_to_consider=individuals_support)
		

		set_of_ballots=set(base_aggregates.keys())
		set_of_ballots_sorted=sorted(set_of_ballots)
		set_of_ballots_indices=set(id_entities_to_indices[x] for x in set_of_ballots) if not ids_replaced else set_of_ballots

		VW,W,disagreement_expected=get_VW_W(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_function)
		

		reliability_full=fast_reliability_with_bounds(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_function)
		CIFULL=[-1.,1.]
		if confidence_interval_quality:
			one_tail_left=False
			one_tail_right=False	
			if quality_measure=='CONFLICTUAL':
				one_tail_left=True
			elif quality_measure=='CONSENSUAL':
				one_tail_right=True
			if COMPUTE_BOOTSTRAPING_CI:
				CIFULL=reliability_CI_KILEM_GWET_BLB(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,set_of_ballots,error=quality_threshold,distance_function=distance_function,one_tail_left=one_tail_left,one_tail_right=one_tail_right)#,one_tail_left=one_tail_left,one_tail_right=one_tail_right)
				print '   '
				print CIFULL ,  reliability_full
				print '   '
			else:
				CIFULL=None
		

		# if False:
		# 	print pattern_printer(group_pat,types_individuals_attributes),reliability_full
		#NEED TO INITIALIZE THIS FUCKING SHIT ONCE TODO
		# print ' '
		# print ' '
		# print minimum_k_valid_approx(VW,W,quality_threshold)
		# raw_input('...')
		enumerator_contexts=enumerator_complex_cbo_init_new_config(considered_items_sorted, contexts_attributes,threshold=support_context_threshold,closed=closure_operator,config_init={'indices':set_of_ballots_indices,'indices_bitset':encode_sup(sorted(set_of_ballots_indices),len(considered_items_sorted)),'CI_PARENT_LB':CIFULL[0] if CIFULL is not None else None,'CI_PARENT_UB':CIFULL[1] if CIFULL is not None else None},initValues=initConfig)
		constants_already_computed=False

		dictionnary_of_already_computed_confidence_interval={len(set_of_ballots):CIFULL,len(set_of_ballots)+1:CIFULL} #todo
		list_of_already_computed_confidence_interval=[len(set_of_ballots),len(set_of_ballots)+1]
		for e_p,e_label,e_config in enumerator_contexts:
			pattern_p_value=0.
			nb_explored+=1
			
			if True or not COMPUTE_BOOTSTRAPING_CI:
				if len(e_config['indices'])==len(set_of_ballots_indices):
					#print 'First Round ! '
					continue
			#print e_label
			context_pat=e_label
			
			entities_context=set(indices_to_id_entities[x] for x in e_config['indices'])  if not ids_replaced else e_config['indices']  #entities_context=set(x[entities_id_attribute] for x in e_config['support'])
			support_size=len(entities_context)
			entities_context_bitset=e_config['indices_bitset']
			LB,reliability_pattern,UB=fast_reliability_with_bounds(marginal_infos,base_aggregates,entities_context,domain_of_possible_outcomes,distance_function=distance_function,support_threshold=support_context_threshold,compute_bounds=True)
			
			####################TEMPORARY############
			reliability_pattern_all_individuals=fast_reliability_with_bounds(marginal_infos_all_users,base_aggregates_all_users,entities_context,domain_of_possible_outcomes,distance_function=distance_function,support_threshold=support_context_threshold,compute_bounds=False)
			####################TEMPORARY############

			
		

			if confidence_interval_quality:
				if False:
					CICONTEXT=(e_config['CI_PARENT_LB'],e_config['CI_PARENT_UB']) 
				else:
					CICONTEXT=dictionnary_of_already_computed_confidence_interval[list_of_already_computed_confidence_interval[bisect_left(list_of_already_computed_confidence_interval,support_size)+1]]
					
				if (CICONTEXT is not None) and PRUNE and CICONTEXT[0]<=LB<=UB<=CICONTEXT[1]:
					# print 'hello'
					e_config['flag']=False
					continue
			else:
				CICONTEXT=[-1.,1.]


			
		
			
			one_tail_left=False
			one_tail_right=False	
			if quality_measure=='CONFLICTUAL':
				quality_pattern=max(reliability_full-reliability_pattern,0)
				UBQUALITY=max(reliability_full-LB,0)
				one_tail_left=True
			elif quality_measure=='CONSENSUAL':
				quality_pattern=max(reliability_pattern-reliability_full,0)
				UBQUALITY=max(UB-reliability_full,0)
				one_tail_right=True
			elif quality_measure=='BOTH':
				quality_pattern=max(max(reliability_full-reliability_pattern,0),max(reliability_pattern-reliability_full,0))#abs(reliability_full-reliability_pattern) #max(reliability_full-reliability_pattern,0)
				UBQUALITY=max(max(reliability_full-LB,0),max(UB-reliability_full,0))



			if confidence_interval_quality:
				
				####EXAMPLE TEMPORARY#########
				CI_APPROX_TO_PRINT=[float('-inf'),float('+inf')]
				if using_approxmations:
					
					#VW,W,disagreement_expected=get_VW_W(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_function)
					#CICONTEXT=CI_approx(VW,W,float(len(entities_context)),quality_threshold,disagreement_expected,one_tail_left=one_tail_left,one_tail_right=one_tail_right)
					
					if not compute_p_value:
						CICONTEXT=CI_approx_second(VW,W,float(len(entities_context)),quality_threshold,disagreement_expected,one_tail_left=one_tail_left,one_tail_right=one_tail_right,constants_already_computed=constants_already_computed)
						
					else:
						moy,std,CICONTEXT=CI_approx_second_with_p_value(VW,W,float(len(entities_context)),quality_threshold,disagreement_expected,one_tail_left=one_tail_left,one_tail_right=one_tail_right,constants_already_computed=constants_already_computed)
						tmp=norm.cdf(reliability_pattern, moy, std) #TIMOTHE #USE STUDENT
						pattern_p_value=2*min(tmp, 1 - tmp)
					constants_already_computed=True
				else:
					#nb_random_sample=5000
					CICONTEXT=reliability_CI_WITHOUT_REPLACEMENT_ON_SETS(marginal_infos,base_aggregates,entities_context,domain_of_possible_outcomes,set_of_ballots_sorted,distance_function=distance_function,error=quality_threshold,one_tail_left=one_tail_left,one_tail_right=one_tail_right,nb_per_bag=nb_random_sample)
				
				if COMPUTE_BOOTSTRAPING_CI:
					CICONTEXT=[min(CIFULL[0],CICONTEXT[0]),max(CIFULL[1],CICONTEXT[1])]
				
				list_of_already_computed_confidence_interval.insert(bisect_left(list_of_already_computed_confidence_interval,support_size),support_size)
				dictionnary_of_already_computed_confidence_interval[support_size]=CICONTEXT
					#CI_APPROX_TO_PRINT=CI_APPROX_TMP
					
				if False:
					print pattern_printer(group_pat,types_individuals_attributes),pattern_printer(context_pat,types_context_attributes),CICONTEXT,CIFULL,reliability_pattern,len(entities_context)
					raw_input('**********')
				
				if False:
					e_config['CI_PARENT_LB']=CICONTEXT[0]
					e_config['CI_PARENT_UB']=CICONTEXT[1]

				# print CICONTEXT[0],LB,UB,CICONTEXT[1]
				# raw_input('***')

				if PRUNE and CICONTEXT[0]<=LB<=UB<=CICONTEXT[1]:
					e_config['flag']=False
					# print 'hello'
					continue
			else:
				if PRUNE and UBQUALITY<quality_threshold:
					e_config['flag']=False
					# print 'hello'
					continue
			interesting_pattern_boolean_indicator=(quality_pattern>=quality_threshold) if not confidence_interval_quality else (reliability_pattern>CICONTEXT[1] or reliability_pattern<CICONTEXT[0])
			
			if interesting_pattern_boolean_indicator:
				#print pattern_printer(group_pat,types_individuals_attributes),pattern_printer(context_pat,types_context_attributes)
				current_individuals_support_bitset=individuals_support_bitset
				current_context_support_bitset=entities_context_bitset
				del_indexes=[]
				del_indexes_append=del_indexes.append
				dominated=False
				if concept_of_generality:
					for k in range(len(lists_of_patterns)):
						entities_support_k_bitset=lists_of_patterns[k][6][1]
						individuals_support_k_bitset=lists_of_patterns[k][6][0]
						if (entities_support_k_bitset&current_context_support_bitset==current_context_support_bitset) & (individuals_support_k_bitset&current_individuals_support_bitset==current_individuals_support_bitset):
							dominated=True
							break
						elif (entities_support_k_bitset&current_context_support_bitset==entities_support_k_bitset) & (individuals_support_k_bitset&current_individuals_support_bitset==individuals_support_k_bitset):
							del_indexes_append(k)
				
					if not dominated:

						del_from_list_by_index(lists_of_patterns,del_indexes)
						lists_of_patterns.append((  (group_pat,context_pat),
													reliability_pattern,
													(len(individuals_support),len(entities_context)),
													quality_pattern,CICONTEXT,
													(individuals_support,entities_context),
													(individuals_support_bitset,entities_context_bitset),
													int(sum(base_aggregates[x]['nb_outcomes'] for x in entities_context)),
													reliability_full,
													reliability_pattern_all_individuals,
													pattern_p_value ) )
						
					#Neet to remove the set of patterns that this patterns cover or do not append the pattern if already a subsuming pattern is in the set
					e_config['flag']=False
				else:
					lists_of_patterns.append((  (group_pat,context_pat),
													reliability_pattern,
													(len(individuals_support),len(entities_context)),
													quality_pattern,CICONTEXT,
													(individuals_support,entities_context),
													(individuals_support_bitset,entities_context_bitset),
													int(sum(base_aggregates[x]['nb_outcomes'] for x in entities_context)),
													reliability_full,
													reliability_pattern_all_individuals,
													pattern_p_value ) )


		#raw_input('....')

	lists_of_patterns=sorted(lists_of_patterns,key=lambda x:x[3],reverse=True)


	lists_of_patterns_to_return=[ 
									{ 
										"id_pattern":id_pattern,
										'pattern_full':(pattern[0],pattern[1]),
										"pattern":(pattern_printer(pattern[0],types_individuals_attributes),pattern_printer(pattern[1],types_context_attributes)),
										"group":pattern_printer(pattern[0],types_individuals_attributes),
										"context":pattern_printer(pattern[1],types_context_attributes),
										"reliability_default":reliability_default,
										"reliability_context_all_individuals":reliability_pattern_all_individuals,
										"reliability_context":reliability_pattern,
										"reliability_ref":reliability_full,
										"quality":quality_pattern,
										"size_group_support":support_size[0],
										"size_context_support":support_size[1],
										"nb_outcomes":nb_outcomes,#sum(base_aggregates[x]['nb_outcomes'] for x in entities_context),
										"confidence_interval":CI,
										"full_group_support":support[0],
										"full_context_support":support[1],
										'pattern_p_value':pattern_p_value,
										'state_intra_agreement':'CONFLICTUAL' if reliability_pattern<reliability_full else 'CONSENSUAL'
									} for  id_pattern,(pattern,reliability_pattern,support_size,quality_pattern,CI,support,support_bitset,nb_outcomes,reliability_full,reliability_pattern_all_individuals,pattern_p_value) in enumerate(lists_of_patterns)
								]
	# for row in lists_of_patterns_to_return:
	# 	print row['pattern'],row['reliability_ref'],row['reliability_context_all_individuals'],row['reliability_context']
	# 	raw_input('...')
	print '...........................................................'
	# for kkk in sorted(dictionnary_of_already_computed_confidence_interval):
	# 	print kkk,dictionnary_of_already_computed_confidence_interval[kkk]
	timespent=time()-timespent
	print 'timespent : ', timespent,'nb patterns', len(lists_of_patterns_to_return)
	print '...........................................................'
	#raw_input('.......')
	STATS['nb_explored']=nb_explored
	STATS['nb_patterns']=nb_patterns
	STATS['timespent_execution']=timespent

	find_exceptional_common_contexts_and_individuals.STATS=STATS

	return lists_of_patterns_to_return


def find_exeptional_common_contexts(considered_items_sorted,contexts_attributes,entities_id_attribute,marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal,support_threshold=10,quality_threshold=0.1,quality_measure='CONFLICTUAL'): #CONSENSUAL, BOTH
	PRUNE=True
	NB_PATTERNS_VISITED=0.
	lists_of_patterns=[]
	types_context_attributes=[x['type'] for x in  contexts_attributes]
	

	indices_to_id_entities=[e[entities_id_attribute] for i,e in enumerate(considered_items_sorted)]

	enumerator_contexts=enumerator_complex_cbo_init_new_config(considered_items_sorted, contexts_attributes,threshold=support_threshold)
	reliability_full=fast_reliability_with_bounds(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_function)
	for e_p,e_label,e_config in enumerator_contexts:
		NB_PATTERNS_VISITED+=1
		context_pat=e_label#pattern_printer(e_label,['themes'])
		
		entities_context=set(indices_to_id_entities[x] for x in e_config['indices'])#entities_context=set(x[entities_id_attribute] for x in e_config['support'])
		entities_context_bitset=e_config['indices_bitset']
			#print len([indices_to_id_entities[x] for x in e_config['indices']]),len(entities_context)
		#print {indices_to_id_entities[x] for x in e_config['indices']} == set(x[entities_id_attribute] for x in e_config['support'])
		#raw_input('...')
		LB,reliability_pattern,UB=fast_reliability_with_bounds(marginal_infos,base_aggregates,entities_context,domain_of_possible_outcomes,distance_function=distance_function,support_threshold=support_threshold,compute_bounds=True)
		
		if quality_measure=='CONFLICTUAL':
			quality_pattern=max(reliability_full-reliability_pattern,0)
			UBQUALITY=max(reliability_full-LB,0)
		elif quality_measure=='CONSENSUAL':
			quality_pattern=max(reliability_pattern-reliability_full,0)
			UBQUALITY=max(UB-reliability_full,0)
		elif quality_measure=='BOTH':
			
			quality_pattern=max(max(reliability_full-reliability_pattern,0),max(reliability_pattern-reliability_full,0))#abs(reliability_full-reliability_pattern) #max(reliability_full-reliability_pattern,0)
			UBQUALITY=max(max(reliability_full-LB,0),max(UB-reliability_full,0))



		if PRUNE and UBQUALITY<quality_threshold:
			e_config['flag']=False
			continue
		if quality_pattern>=quality_threshold:
			#lists_of_patterns.append((context_pat,reliability_pattern,len(entities_context),quality_pattern))
			current_support=entities_context
			current_support_bitset=entities_context_bitset
			del_indexes=[]
			del_indexes_append=del_indexes.append
			dominated=False
			for k in range(len(lists_of_patterns)):
				support_k=lists_of_patterns[k][5]
				support_k_bitset=lists_of_patterns[k][6]
				#if (support_k>=current_support):
				if (support_k_bitset&current_support_bitset==current_support_bitset):
					dominated=True
					break
				#elif (support_k_bitset<=current_support_bitset):
				elif (support_k_bitset&current_support_bitset==support_k_bitset):
					del_indexes_append(k)
			
			if not dominated:
				del_from_list_by_index(lists_of_patterns,del_indexes)
				lists_of_patterns.append((context_pat,reliability_pattern,len(entities_context),quality_pattern,[-1.,1.],entities_context,entities_context_bitset))

			#Neet to remove the set of patterns that this patterns cover or do not append the pattern if already a subsuming pattern is in the set
			e_config['flag']=False

		
	lists_of_patterns=sorted(lists_of_patterns,key=lambda x:x[3],reverse=True)


	lists_of_patterns_to_return=[ 
									{ 
										"id_pattern":id_pattern,
										"pattern":pattern_printer(pattern,types_context_attributes),
										"reliability_default":reliability_full,
										"reliability_context":reliability_pattern,
										"reliability_context_all_individuals":reliability_pattern,
										"reliability_ref":reliability_full,
										"quality":quality_pattern,
										"size_group_support":int(marginal_infos['nb_raters']),
										"size_context_support":support_entities_size,
										
										"nb_outcomes":sum(base_aggregates[x]['nb_outcomes'] for x in entities_context),
										"confidence_interval":CI,

										"full_context_support":entities_context,

										'state_intra_agreement':'CONFLICTUAL' if reliability_pattern<reliability_full else 'CONSENSUAL'
									} for  id_pattern,(pattern,reliability_pattern,support_entities_size,quality_pattern,CI,entities_context,entities_context_bitset) in enumerate(lists_of_patterns)
								]


	print 'NB_PATTERNS_VISITED : ', NB_PATTERNS_VISITED,'NB_PATTERNS_FOUND : ',len(lists_of_patterns_to_return)
	#raw_input('*****')
	return lists_of_patterns_to_return

#@profile
#find_exeptional_contexts_with_CIs
def find_exeptional_contexts_with_CIs_BOTH(considered_items_sorted,contexts_attributes,entities_id_attribute,marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal,support_threshold=10,quality_threshold=0,quality_measure='CONFLICTUAL'):
	PRUNE=True
	NB_PATTERNS_VISITED=0.
	lists_of_patterns=[]
	types_context_attributes=[x['type'] for x in  contexts_attributes]
	
	reliability_full=fast_reliability_with_bounds(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_function)
	CIFULL=reliability_CI_KILEM_GWET_BLB(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,set_of_ballots,distance_function=distance_function)
	enumerator_contexts=enumerator_complex_cbo_init_new_config(considered_items_sorted, contexts_attributes,threshold=support_threshold,config_init={'CI_PARENT_LB':CIFULL[0],'CI_PARENT_UB':CIFULL[1]})
	# print CIFULL
	# print reliability_full
	indices_to_id_entities=[e[entities_id_attribute] for i,e in enumerate(considered_items_sorted)]
	




	for e_p,e_label,e_config in enumerator_contexts:
		NB_PATTERNS_VISITED+=1
		context_pat=e_label
		#entities_context=set(x[entities_id_attribute] for x in e_config['support'])
		entities_context=set(indices_to_id_entities[x] for x in e_config['indices'])
		LB,reliability_pattern,UB=fast_reliability_with_bounds(marginal_infos,base_aggregates,entities_context,domain_of_possible_outcomes,distance_function=distance_function,support_threshold=support_threshold,compute_bounds=True)
		
		CICONTEXT=(e_config['CI_PARENT_LB'],e_config['CI_PARENT_UB'])
		
		if PRUNE and CICONTEXT[0]<=LB<=UB<=CICONTEXT[1]:
			e_config['flag']=False
			continue
		if CICONTEXT[0]<=reliability_pattern<=CICONTEXT[1]:
			continue
		one_tail_left=False
		one_tail_right=False
		if quality_measure=='CONFLICTUAL':
			one_tail_left=True
		elif quality_measure=='CONSENSUAL':
			one_tail_right=True
		support_size=len(entities_context)
		CICONTEXT=reliability_CI_WITHOUT_REPLACEMENT_ON_SETS(marginal_infos,base_aggregates,entities_context,domain_of_possible_outcomes,set_of_ballots,distance_function=distance_function,error=quality_threshold,one_tail_left=one_tail_left,one_tail_right=one_tail_right)



		
		e_config['CI_PARENT_LB']=CICONTEXT[0]
		e_config['CI_PARENT_UB']=CICONTEXT[1]
		
		if False:
			print ''
			print '------------------------------------------------------------'
			print 'Size Pattern : ',  len(entities_context)
			print 'PATTERN Bounds : ', (LB,UB)
			print 'Confidence interval : ',CICONTEXT
			print 'Pattern quality : ', reliability_pattern, '\t',CICONTEXT
			print '------------------------------------------------------------'
			print ''

		quality_pattern=abs(reliability_pattern-reliability_full)
		#UBQUALITY=((UB-reliability_full)**2)**(0.5)
		#LBQUALITY=((reliability_full-LB)**2)**(0.5)
		#UBB=max(LBQUALITY,UBQUALITY)
		
		if PRUNE and CICONTEXT[0]<=LB<=UB<=CICONTEXT[1]:
			e_config['flag']=False
			continue


		if reliability_pattern>CICONTEXT[1] or reliability_pattern<CICONTEXT[0] :
			

			#################################################
			current_support=entities_context
			del_indexes=[]
			del_indexes_append=del_indexes.append
			dominated=False
			for k in range(len(lists_of_patterns)):
				support_k=lists_of_patterns[k][5]
				if (support_k>=current_support):
					dominated=True
					break
				elif (support_k<=current_support):
					del_indexes_append(k)
			
			if not dominated:
				del_from_list_by_index(lists_of_patterns,del_indexes)
				lists_of_patterns.append((context_pat,reliability_pattern,len(entities_context),quality_pattern,CICONTEXT,entities_context))
			#Neet to remove the set of patterns that this patterns cover or do not append the pattern if already a subsuming pattern is in the set
			e_config['flag']=False
			##################################################
		
	



	lists_of_patterns=sorted(lists_of_patterns,key=lambda x:x[3],reverse=True)


	lists_of_patterns_to_return=[ 
									{ 
										"id_pattern":id_pattern,
										"pattern":pattern_printer(pattern,types_context_attributes),
										"reliability_context":reliability_pattern,
										"reliability_ref":reliability_full,
										"quality":quality_pattern,
										"size_context_support":support_entities_size,
										"nb_outcomes":sum(base_aggregates[x]['nb_outcomes'] for x in entities_context),
										"confidence_interval":CI,
										"full_context_support":entities_context,
										'state_intra_agreement':'CONFLICTUAL' if reliability_pattern<reliability_full else 'CONSENSUAL'
									} for  id_pattern,(pattern,reliability_pattern,support_entities_size,quality_pattern,CI,entities_context) in enumerate(lists_of_patterns)
								]


	print 'NB_PATTERNS_VISITED : ', NB_PATTERNS_VISITED
	return lists_of_patterns_to_return


def jaccard(s1,s2):
	return float(len(s1&s2))/len(s1|s2)

def get_top_k_div_from_a_pattern_set(patterns,threshold_sim=0.6,k=1000): #patterns = [(p,rel,len_sup,qual,ci,sup)]
	
	
	returned_patterns=[]
	sorted_patterns=sorted(patterns,key=lambda x:x[3],reverse=True)
	tp=sorted_patterns[0]
	returned_patterns.append(tp)
	returned_patterns_indice={0}
	while 1:
		if len(returned_patterns)==k:
			break

		found_yet=False
		
		for i,p in enumerate(sorted_patterns):
			if i in returned_patterns_indice:
				continue
			sup=p[5]
			if all(jaccard(sup,supc)<=threshold_sim for _,_,_,_,_,supc in returned_patterns):
				found_yet=True
				#t_p=(p,sup,supbitset,qual)
				returned_patterns.append(p)
				returned_patterns_indice|={i}
				break
		
		if not found_yet:
			break

		
	return returned_patterns

def get_top_k_div_from_a_pattern_set_cumulative_quality(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,patterns,distance_function=distance_nominal,k=1000,quality_threshold=0.1): #patterns = [(p,rel,len_sup,qual,ci,sup)]
	print "nb initial set : ",len(patterns)
	full_reliability=reliability(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal)
	returned_patterns=[]
	sorted_patterns=sorted(patterns,key=lambda x:x[3],reverse=True)
	tp=sorted_patterns[0]
	returned_patterns.append(tp)
	returned_patterns_indice={0}
	returned_pattern_support= set(tp[5])
	current_quality=tp[3]
	
	while 1:
		if len(returned_patterns)==k:
			break

		found_yet=False
		best_marginal_contribution=quality_threshold
		best_pattern_yet_index=None
		for i,p in enumerate(sorted_patterns):
			if i in returned_patterns_indice:
				continue
			
			sup=p[5]
			sup_to_test=returned_pattern_support|sup
			quality_union=max(full_reliability-reliability(marginal_infos,base_aggregates,sup_to_test,domain_of_possible_outcomes,distance_function=distance_nominal),0.)
			#print quality_union,current_quality
			#raw_input('.....')
			marginal_contribution_with_the_selected_pattern=quality_union# - current_quality
			if marginal_contribution_with_the_selected_pattern>best_marginal_contribution:
				found_yet=True
				best_pattern_yet_index=i
				best_marginal_contribution=marginal_contribution_with_the_selected_pattern


			
		
		if not found_yet:
			break
		else:
			returned_patterns.append(sorted_patterns[best_pattern_yet_index])
			returned_patterns_indice|={best_pattern_yet_index}
			current_quality=best_marginal_contribution
			returned_pattern_support=returned_pattern_support|sorted_patterns[best_pattern_yet_index][5]


	print "nb final set : ",len(returned_patterns)	
	print "quality union : ",  current_quality
	return returned_patterns
		#jaccard(p[5],



def weighted_average(subset_values):
	sum_v=0.
	sum_w=0.
	for x in subset_values:
		sum_v+=x[0]
		sum_w+=x[1]
	return sum_v/sum_w

def weighted_average_weird_correction(subset_values):
	sum_v=0.
	sum_w=1.
	for x in subset_values:
		sum_v+=x[0]
		sum_w+=x[1]
	return sum_v/sum_w

def newton_optimized(values,k,bottom=False):
	T=values[:k]
	prec_avg=weighted_average(T)
	while True:
		if bottom:
			T_next=argpartsort(([vj-prec_avg*wj for (vj,wj) in values]),k)[:k]
		else:
			T_next=argpartsort(([prec_avg*wj-vj for (vj,wj) in values]),k)[:k]

		now_avg=weighted_average(((values[j][0],values[j][1]) for j in T_next))
		if abs(prec_avg - now_avg)<10**(-9):
			break
		prec_avg=now_avg
	return now_avg

def newton_optimized_weird_correction(values,k,bottom=False):
	T=values[:k]
	prec_avg=weighted_average_weird_correction(T)
	while True:
		if bottom:
			T_next=argpartsort(([vj-prec_avg*wj for (vj,wj) in values]),k)[:k]
		else:
			T_next=argpartsort(([prec_avg*wj-vj for (vj,wj) in values]),k)[:k]

		now_avg=weighted_average_weird_correction(((values[j][0],values[j][1]) for j in T_next))
		if abs(prec_avg - now_avg)<10**(-9):
			break
		prec_avg=now_avg
	return now_avg


def epstein_max_and_min(values,k,bottom=False):
	#print 'I AM USED BITCH'
	if not bottom:
		to_ret=random_epstein_exact(values,k)
		#print to_ret
		return to_ret
	else:
		xmax=max([x/float(y) for x,y in values])#max([y for x,y in values])
		values=[(y * xmax - x,y) for x,y in values]
		to_ret=xmax-random_epstein_exact(values,k)

		# values=[(0. - x,y) for x,y in values]
		# to_ret=0.-random_epstein_exact(values,k)
		# print to_ret
		return to_ret

def epstein_max_and_min_weird_correction(values,k,bottom=False):
	if not bottom:
		return random_epstein_exact_weird_correction(values,k)
	else:
		xmax=max([x/float(y) for x,y in values])#max([y for x,y in values])
		values=[(y * xmax - x,y) for x,y in values]
		return xmax-random_epstein_exact_weird_correction(values,k)

def exact_computing(values,k,bottom=False):
	wavg=0.
	if bottom:
		wavg=float('inf')
	for elems in combinations(range(len(values)), k):
		values_selected=[values[x] for x in elems]
		wavg_current=weighted_average(values_selected)
		if wavg_current>=wavg and not bottom or wavg_current<=wavg and bottom:
			wavg=wavg_current
	return wavg

def exact_computing_weird_correction(values,k,bottom=False):
	wavg=0.
	if bottom:
		wavg=float('inf')
	for elems in combinations(range(len(values)), k):
		values_selected=[values[x] for x in elems]
		wavg_current=weighted_average_weird_correction(values_selected)
		if wavg_current>=wavg and not bottom or wavg_current<=wavg and bottom:
			wavg=wavg_current
	return wavg




# def distance_nominal(o1,o2,domain_of_possible_outcomes=None):
# 	return int(o1!=o2)

# def distance_categorical(o1,o2,domain_of_possible_outcomes=None):
# 	return int(o1!=o2)

# def distance_ordinal(o1,o2,domain_of_possible_outcomes=None): #STILL NEED TO ADD INFO ABOUT THE MAXIMUM
# 	return 0.

ALGORITHM_TO_USE_DICTIONNARY={
	"COMMON":find_exeptional_common_contexts,
	"COMMON_PEERS":find_exceptional_common_contexts_and_individuals,
	"P_VALUE_PEERS":find_exceptional_common_contexts_and_individuals,
	"P_VALUE":find_exeptional_contexts_with_CIs_BOTH
}



QUALITY_MEASURES=['CONFLICTUAL','CONSENSUAL','BOTH']
OUTCOMES_DISTANCES_FUNCTION_TO_USE={
	"categorical":distance_categorical,
	"ordinal":distance_ordinal_simple,
	"ordinal_complex":distance_ordinal_complex,


}




def Depict_input_config(json_config,performance_test=False,compute_empirical_distribution=False,replace_ids=True,first_run=True,no_additional_files=False):
	timespent_init=time()
	attributes_context=[]
	attributes_individuals=[]
	entities_file_path=json_config["objects_file"]
	individuals_file_path=json_config["individuals_file"]
	outcomes_file=json_config["reviews_file"]
	delimiter=json_config.get("delimiter","\t")

	nb_items=json_config.get("nb_objects",float('inf'))
	nb_individuals=json_config.get("nb_individuals",float('inf'))


	array_attrs=json_config.get("arrayHeader",[])
	numbers_header=json_config.get("numericHeader",[])


	nb_items_individuals=json_config.get("nb_items_individuals",float('inf'))
	nb_items_entities=json_config.get("nb_items_entities",float('inf'))
	objects_Scope=json_config.get("objects_scope",[]) #all dataset on input
	contexts_scope=json_config.get("contexts_scope",[]) #where to mind the business of mining while considering all objects scope as a reference context 
	inidividuals_scope=json_config.get("individuals_scope",[])
	
	ratings_to_remove=set(json_config.get("ratings_to_remove",[]))
	
	nb_attributes_entities=json_config.get("nb_attributes_entities",1000)
	nb_attributes_individuals=json_config.get("nb_attributes_individuals",1000)
	

	description_attributes_objects=json_config.get("description_attributes_objects",[])[:nb_attributes_entities]
	description_attributes_individuals=json_config.get("description_attributes_individuals",[])[:nb_attributes_individuals]

	attributes_to_consider=[x for x,y in description_attributes_objects]+[x for x,y in description_attributes_individuals]


	threshold_objects=json_config.get("threshold_objects",1)
	threshold_individuals=json_config.get("threshold_individuals",1)
	threshold_quality=json_config.get("threshold_quality",0.)
	outcome_distance_function=json_config.get("outcome_distance_function","categorical")

		
	description_attributes_objects=[{'name':x, 'type':y} for x,y in description_attributes_objects]
	description_attributes_individuals=[{'name':x, 'type':y} for x,y in description_attributes_individuals]
	############ALGORITHMS PARAMETERS############

	METHOD_TO_USE=json_config.get("algorithm","COMMON") #"CONFLICTUAL"|"CONSENSUAL"|"BOTH"|"P_VALUE"|"P_VALUE_CONFLICTUAL"|"P_VALUE_CONSENSUAL"
	QUALITY_MEASURE=json_config.get("quality_measure","CONFLICTUAL") #"CONFLICTUAL"|"CONSENSUAL"|"BOTH"|"P_VALUE"|"P_VALUE_CONFLICTUAL"|"P_VALUE_CONSENSUAL"
	distance_function=OUTCOMES_DISTANCES_FUNCTION_TO_USE[outcome_distance_function]
	#############################################

	##########
	similarity_matrix=json_config.get("similarity_matrix",False)
	key_to_print=json_config.get("key_to_print",None)
	########


	closure_operator=json_config.get("closure_operator",True)
	pruning=json_config.get("pruning",True)
	using_approxmations=json_config.get("using_approxmations",False)
	nb_random_sample=json_config.get("nb_random_sample",5000)

	compute_bootstrap_ci=json_config.get("compute_bootstrap_ci",False)
	compute_p_value=json_config.get("compute_p_value",False)
	# print compute_p_value
	# raw_input('....')
	#################################################

	results_destination=json_config.get("results_destination","./results.csv")
	detailed_results_destination=json_config.get("detailed_results_destination",".//Results//")

	concept_of_generality=not json_config.get("no_concept_of_generality",False)
	#################################################


	if False:
		weights_individuals=None #which can also represent the number of individual that comprise a pre-aggregated group
		method_aggregation_outcome="-" #default but can be VECTOR_VALUES  for a distribution
		vector_of_outcome=["full_outcomes"]
	else:
		weights_individuals=None #which can also represent the number of individual that comprise a pre-aggregated group
		method_aggregation_outcome="-" #default but can be VECTOR_VALUES  for a distribution
		vector_of_outcome=None
	############1) PROCESS BEHAVIORAL DATA###########
	entities_metadata, \
	individuals_metadata, \
	all_individuals_to_entities_outcomes, \
	all_entities_to_individuals_outcomes, \
	domain_of_possible_outcomes, \
	outcomes_considered, \
	entities_id_attribute, \
	individuals_id_attribute, \
	considered_items_sorted, \
	considered_users_1_sorted, \
	considered_users_2_sorted, \
	nb_outcome_considered = \
	process_outcome_dataset(
		entities_file_path,
		individuals_file_path,
		outcomes_file,
		numeric_attrs=numbers_header,
		array_attrs=array_attrs,
		outcome_attrs=vector_of_outcome,
		method_aggregation_outcome=method_aggregation_outcome,#'SYMBOLIC_MAJORITY',
		itemsScope=objects_Scope,
		users_1_Scope=inidividuals_scope,
		users_2_Scope=inidividuals_scope,
		ratings_to_remove=ratings_to_remove,
		nb_items=nb_items,
		nb_individuals=nb_individuals,
		attributes_to_consider=attributes_to_consider,
		nb_items_entities=nb_items_entities,
		nb_items_individuals=nb_items_individuals, 
		hmt_to_itemset=False,
		delimiter=delimiter,
		replace_ids=replace_ids
	)
	###########################################
	if False:
		domain_of_possible_outcomes=sorted(domain_of_possible_outcomes)
		base_aggregates,marginal_infos=pre_compute_base_aggregates_for_kripendorff(all_entities_to_individuals_outcomes,domain_of_possible_outcomes,all_individuals_to_entities_outcomes,distance_function=distance_function)
		#domain_of_possible_outcomes=sorted(domain_of_possible_outcomes)
		entities_votes_details_distribution={e:[base_aggregates[e][o] for o in sorted(domain_of_possible_outcomes)] for e in base_aggregates}
		entities_votes_majorities_decision={e:domain_of_possible_outcomes[entities_votes_details_distribution[e].index(max(entities_votes_details_distribution[e]))] for e in entities_votes_details_distribution}
		print len(all_entities_to_individuals_outcomes)
		
		header_to_write=['vote_id','congress','year','description','vote_id_in_congress','billtype','topic','MAJORITY_VOTE','MAJORITY_VOTE_DISTRIBUTION']
		##TODO

		# for e in sorted(entities_votes_details_distribution):
			
		# 	print considered_items_sorted[e],e,entities_votes_details_distribution[e],entities_votes_majorities_decision[e]
		# 	raw_input('....')

		for row in considered_items_sorted:
			id_row=row[entities_id_attribute]
			row['MAJORITY_VOTE_DISTRIBUTION'] =entities_votes_details_distribution[id_row]
			row['MAJORITY_VOTE'] =entities_votes_majorities_decision[id_row]
		
			
		writeCSVwithHeader(considered_items_sorted,'items_NEW_CHUS.csv',header_to_write,delimiter='\t')	
		raw_input('....')
		raw_input('....')
	###########################################
	
	############1) PROCESS BEHAVIORAL DATA###########

	##TMP##
	# indices_to_id_individuals=[u[individuals_id_attribute] for i,u in enumerate(considered_users_1_sorted)]
	# enumerator_individuals=enumerator_complex_cbo_init_new_config(considered_users_1_sorted, description_attributes_individuals,threshold=threshold_individuals)
	# for i_p,i_label,i_config in enumerator_individuals:
		
	# 	individuals_support=set(indices_to_id_individuals[x] for x in i_config['indices'])
	# 	print bin(i_config['indices_bitset'])
	# 	print i_p,len(individuals_support)/float(len(indices_to_id_individuals))
	# 	raw_input('....')
	##TMP##


	############2) Build the base aggregate for Fast Computation################
	domain_of_possible_outcomes=sorted(domain_of_possible_outcomes)
	print "Finished processing the files (a)..."
	time_start=time()
	base_aggregates,marginal_infos=pre_compute_base_aggregates_for_kripendorff(all_entities_to_individuals_outcomes,domain_of_possible_outcomes,all_individuals_to_entities_outcomes,distance_function=distance_function)
	print time()-time_start
	print "Precomputing basic aggregate value to speed up the algorithm (b)..."
	set_of_ballots=all_entities_to_individuals_outcomes.viewkeys()
	reliability_reference=fast_reliability_with_bounds(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_function)
	print 'REFERENCE RELIABILITY (USUAL INTER-AGREEMENT) : ', reliability_reference
	#raw_input('...........')
	############2) Build the base aggregate for Fast Computation################

	############3) Look for interesting Patterns################################
	
	
	if  METHOD_TO_USE[-5:]=="PEERS":
		print 'Peers method invoked HALILUJAH ! ...'
		confidence_interval_quality=False
		if METHOD_TO_USE=='P_VALUE_PEERS':
			confidence_interval_quality=True
		


		#############################FIRST CONSIDERATION#################
		if compute_empirical_distribution:
			for_empirical_distributions_generation=True
			return_distribution=False
			number_of_distributions=400
			if for_empirical_distributions_generation:
				return_distribution=True
				number_of_distributions=5
			print 'EMPIRICAL DISTRIBUTION !!!' 
			first_run_my_boy=True
			VW,W,disagreement_expected=get_VW_W(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_function) #get_vw_w_all_ready
			
			#VW_PRE_COMPUTED,W_PRE_COMPUTED,xxx=get_vw_w_all_ready(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_function)
			

			dataset_to_write_CI_APPROX_TMP=CI_approx_empirical_comparison(VW,W,threshold_quality,disagreement_expected,nb_random_sample=nb_random_sample,one_tail_left=False,one_tail_right=False,return_distribution=return_distribution,number_of_distributions=number_of_distributions)
			filename, file_extension = os.path.splitext(results_destination)

			all_seen_dists=[]
			all_seen_dists_name=[]
			for row in dataset_to_write_CI_APPROX_TMP:
				if for_empirical_distributions_generation:
					from plotter.perfPlotter import plotHistograms_to_test_edf
					print len(row["empirical_distribution"])
					all_seen_dists.append(row["empirical_distribution"])
					all_seen_dists_name.append("k="+str(row["sample_size"]).zfill(4)+";  CI=" +  "[" + ("%.3f"%row["confidence_interval_empirical"][0]) +","+ ("%.3f"%row["confidence_interval_empirical"][1])+"]")
					
				writeCSVwithHeader([row],filename+'_'+str(threshold_quality)+'_empirical_taylor_approx.csv',['alpha','dataset_size','minimum_threshold_alpha','sample_size','confidence_interval_taylor','confidence_interval_empirical','confidence_interval_empirical_percentile','error','error_empirical','normal_distribution_pvalue','follows_normal'],delimiter='\t',flagWriteHeader=first_run_my_boy)
				first_run_my_boy=False
			if for_empirical_distributions_generation:
				plotHistograms_to_test_edf(all_seen_dists,all_seen_dists_name,manydistributions=True)
			return []
			#raw_input('...')

		#############################FIRST CONSIDERATION#################

		# lists_of_patterns=ALGORITHM_TO_USE_DICTIONNARY[METHOD_TO_USE](considered_users_1_sorted,
		# 											 considered_items_sorted,
		# 											 description_attributes_individuals,
		# 											 description_attributes_objects,
		# 											 individuals_id_attribute,
		# 											 entities_id_attribute,
		# 											 all_entities_to_individuals_outcomes,
		# 											 all_individuals_to_entities_outcomes,
		# 											 domain_of_possible_outcomes,
		# 											 distance_function=distance_function,
		# 											 support_context_threshold=threshold_objects,
		# 											 support_individuals_threshold=threshold_individuals,
		# 											 quality_threshold=threshold_quality,
		# 											 quality_measure=QUALITY_MEASURE,
		# 											 confidence_interval_quality=confidence_interval_quality)
		timespent_init=time()-timespent_init
		lists_of_patterns=find_exceptional_common_contexts_and_individuals(considered_users_1_sorted,
													 considered_items_sorted,
													 description_attributes_individuals,
													 description_attributes_objects,
													 individuals_id_attribute,
													 entities_id_attribute,
													 all_entities_to_individuals_outcomes,
													 all_individuals_to_entities_outcomes,
													 domain_of_possible_outcomes,
													 distance_function=distance_function,
													 support_context_threshold=threshold_objects,
													 support_individuals_threshold=threshold_individuals,
													 quality_threshold=threshold_quality,
													 quality_measure=QUALITY_MEASURE,
													 confidence_interval_quality=confidence_interval_quality,
													 closure_operator=closure_operator,
													 pruning=pruning,
													 using_approxmations=using_approxmations,
													 nb_random_sample=nb_random_sample,
													 compute_bootstrap_ci=compute_bootstrap_ci,
													 compute_p_value=compute_p_value,
													 concept_of_generality=concept_of_generality,
													 replace_ids=replace_ids
													 )

		# print process_outcome_dataset.STATS
		# raw_input('...')
		find_exceptional_common_contexts_and_individuals.STATS['compute_bootstrap_ci'] = compute_bootstrap_ci
		find_exceptional_common_contexts_and_individuals.STATS['nb_individuals_attributes_items'] = process_outcome_dataset.STATS['nb_items_individuals']
		find_exceptional_common_contexts_and_individuals.STATS['nb_contexts_attributes_items'] = process_outcome_dataset.STATS['nb_items_entities']
			
		find_exceptional_common_contexts_and_individuals.STATS['nb_individuals_attributes'] = len(description_attributes_individuals)
		find_exceptional_common_contexts_and_individuals.STATS['nb_contexts_attributes'] = len(description_attributes_objects)

		find_exceptional_common_contexts_and_individuals.STATS['nb_patterns'] = len(lists_of_patterns)
		find_exceptional_common_contexts_and_individuals.STATS['distance_function']=outcome_distance_function
		find_exceptional_common_contexts_and_individuals.STATS['timespent_init']=timespent_init
		find_exceptional_common_contexts_and_individuals.STATS['timespent_total']=timespent_init+find_exceptional_common_contexts_and_individuals.STATS['timespent_execution']
		
		print '-----------------------------STATS-----------------------------'
		for key in find_exceptional_common_contexts_and_individuals.STATS:
			print key,': \t ',find_exceptional_common_contexts_and_individuals.STATS[key]

		print '-----------------------------STATS-----------------------------'	
		if performance_test:
			filename, file_extension = os.path.splitext(results_destination)
			HEADER=['compute_bootstrap_ci','nb_entities','nb_individuals','nb_individuals_attributes','nb_individuals_attributes_items','nb_contexts_attributes','nb_contexts_attributes_items','support_individuals_threshold','support_context_threshold','quality_threshold','closure_operator','pruning','using_approxmations','nb_random_sample','distance_function','timespent_execution','timespent_init','timespent_total','nb_explored','nb_patterns']
			writeCSVwithHeader([find_exceptional_common_contexts_and_individuals.STATS],filename+'_performance_test.csv',HEADER,delimiter='\t',flagWriteHeader=first_run)
			return lists_of_patterns
	else:
		lists_of_patterns=ALGORITHM_TO_USE_DICTIONNARY[METHOD_TO_USE](considered_items_sorted,description_attributes_objects,entities_id_attribute,marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,support_threshold=threshold_objects,quality_threshold=threshold_quality,quality_measure=QUALITY_MEASURE)




	return Depict_output(considered_users_1_sorted,considered_items_sorted,individuals_id_attribute,entities_id_attribute,description_attributes_objects,description_attributes_individuals,all_entities_to_individuals_outcomes,all_individuals_to_entities_outcomes,lists_of_patterns,distance_function,base_aggregates,marginal_infos,domain_of_possible_outcomes,similarity_matrix,key_to_print,results_destination,detailed_results_destination,replace_ids,no_additional_files)



	#,x['confidence_interval'],x['full_context_support']
		#raw_input('......')

	############3) Look for interesting Patterns################################


#############################

def compute_similarity_matrix(considered_users_1_sorted,individuals_id_attribute,all_entities_to_individuals_outcomes,all_individuals_to_entities_outcomes,full_group_support,full_context_support,distance_function=distance_nominal,marginal_infos=None,domain_of_possible_outcomes=None,key_to_print=None):
	if key_to_print is not None:
		list_mapping={row[individuals_id_attribute]:('-'.join(unicodedata.normalize('NFD', unicode(str(row[kkk]),'iso-8859-1')).encode('ascii', 'ignore') for kkk in key_to_print)).replace('_','-') for row in considered_users_1_sorted}
	transforming_ids=str if key_to_print is None else lambda x:list_mapping[x]+'-'+str(x)

	sorted_full_group_support=sorted(full_group_support)
	sim_dict_ref={transforming_ids(id_ind_1):{transforming_ids(id_ind_2):float('NaN') for id_ind_2 in sorted_full_group_support} for id_ind_1 in sorted_full_group_support}
	sim_dict_context={transforming_ids(id_ind_1):{transforming_ids(id_ind_2):float('NaN') for id_ind_2 in sorted_full_group_support} for id_ind_1 in sorted_full_group_support}
	nb_total=float(len(sorted_full_group_support)+(len(sorted_full_group_support)*(len(sorted_full_group_support)-1))/2.)
	#stdout.write('%s\r' % ('Percentage Done : ' + ('%.2f'%((count_i_p/float(nb_total))*100))+ '%'));stdout.flush();
	count_i_p=0
	for id_ind_1 in sorted_full_group_support:
		 for id_ind_2 in sorted_full_group_support:
		 	count_i_p+=1
		 	stdout.write('%s\r' % ('Percentage Done : ' + ('%.2f'%((count_i_p/float(nb_total))*100))+ '%'));stdout.flush();
		 	if True:
			 	#print id_ind_1,id_ind_2 
			 	id_ind_1_int=transforming_ids(id_ind_1)
			 	id_ind_2_int=transforming_ids(id_ind_2)
			 	all_concerned=(set(all_individuals_to_entities_outcomes[id_ind_1])) & set(all_individuals_to_entities_outcomes[id_ind_2])
				context_concerned=all_concerned & set(full_context_support)

				if len(all_concerned)>0:
					sim_dict_ref[id_ind_1_int][id_ind_2_int]=(sum([(1.-distance_function(all_individuals_to_entities_outcomes[id_ind_1][e],all_individuals_to_entities_outcomes[id_ind_2][e],domain_of_possible_outcomes=domain_of_possible_outcomes,marginal_infos=marginal_infos)) for e in all_concerned])/float(len(all_concerned)))
					sim_dict_ref[id_ind_2_int][id_ind_1_int]=sim_dict_ref[id_ind_1_int][id_ind_2_int]
					if len(context_concerned)>0:
						sim_dict_context[id_ind_1_int][id_ind_2_int]=(sum([(1.-distance_function(all_individuals_to_entities_outcomes[id_ind_1][e],all_individuals_to_entities_outcomes[id_ind_2][e],domain_of_possible_outcomes=domain_of_possible_outcomes,marginal_infos=marginal_infos)) for e in context_concerned])/float(len(context_concerned)))
						sim_dict_context[id_ind_2_int][id_ind_1_int]=sim_dict_context[id_ind_1_int][id_ind_2_int]
						#print sim_dict_ref[id_ind_1][id_ind_2]
						#raw_input('...')
			if id_ind_2>id_ind_1:
				break
	return sim_dict_ref,sim_dict_context


############################


def Depict_output(considered_users_1_sorted,considered_items_sorted,individuals_id_attribute,entities_id_attribute,description_attributes_objects,description_attributes_individuals,all_entities_to_individuals_outcomes,all_individuals_to_entities_outcomes,lists_of_patterns,distance_function=distance_nominal,base_aggregates=None,marginal_infos=None,domain_of_possible_outcomes=None,similarity_matrix=False,key_to_print=None,results_destination=None,detailed_results_destination=None,replace_ids=True,no_additional_files=False):
	VERBOSE=False
	ADD_SIMILARITY_MATRIX=similarity_matrix
	if VERBOSE:
		print 'NB PATTERNS FOUND : ',len(lists_of_patterns)
		for x in lists_of_patterns:
			print x['id_pattern'],x['pattern'],x['reliability_context'],x['reliability_ref'],x['quality'],x['size_context_support'],x['nb_outcomes'],x['state_intra_agreement'],x['confidence_interval'],x['full_context_support'],x['pattern_p_value']
	if detailed_results_destination is not None:
		if not os.path.exists(detailed_results_destination):
			os.makedirs(detailed_results_destination)
		else:
			if False:
				shutil.rmtree(detailed_results_destination)
				os.makedirs(detailed_results_destination)
	
	indices_to_id_individuals=[u[individuals_id_attribute] for i,u in enumerate(considered_users_1_sorted)]
	id_individuals_to_indices={u:i for i,u in enumerate(indices_to_id_individuals)}
	indices_to_id_entities=[e[entities_id_attribute] for i,e in enumerate(considered_items_sorted)]
	id_entities_to_indices={e:i for i,e in enumerate(indices_to_id_entities)}

	if results_destination is not None:
		header=['id_pattern','pattern','size_group_support','size_context_support','nb_outcomes' , 'reliability_default','reliability_ref','reliability_context_all_individuals','reliability_context','quality', 'confidence_interval','pattern_p_value','state_intra_agreement']
		writeCSVwithHeader(lists_of_patterns,results_destination,header,delimiter='\t')
		lists_of_patterns_reduce_precision=[]
		for row in lists_of_patterns:
			newrow=row.copy()
			newrow["reliability_ref"]=float("%.2f"%newrow["reliability_ref"])
			newrow["reliability_context"]=float("%.2f"%newrow["reliability_context"])
			newrow["pattern_p_value"]= "<0.0001" if   newrow["pattern_p_value"] <0.0001 else "<0.001" if   newrow["pattern_p_value"] <0.001 else "<0.01" if   newrow["pattern_p_value"] <0.01 else "<0.05"
			lists_of_patterns_reduce_precision.append(newrow)

		writeCSVwithHeader(lists_of_patterns_reduce_precision,os.path.splitext(results_destination)[0]+"_for_latex.csv",["id_pattern","group","context","reliability_ref","reliability_context","pattern_p_value","state_intra_agreement"],delimiter='\t')
		if True:
			print '\t'.join(header)
			domain_of_possible_outcomes_sorted=sorted(domain_of_possible_outcomes)
			for row in lists_of_patterns:
				print '\t'.join(str('%.2f'%row[x]) if type(row[x]) is float else str(row[x]) for x in header)
				
				#############
				if not no_additional_files and detailed_results_destination is not None:
					data_items_to_print=[]
					data_groups_to_print=[]
					data_groups_outcomes_to_print=[]
					#print "################PATTERN" + str(row['id_pattern'])+"############"
					for i in row['full_context_support']:
						
						data_items_to_print.append(considered_items_sorted[id_entities_to_indices[i]].copy())
						data_items_to_print[-1]['krippendorff_alpha']=fast_reliability_with_bounds(marginal_infos,base_aggregates,{i},domain_of_possible_outcomes,distance_function=distance_function)
						data_items_to_print[-1]['nb_voters']=len(all_entities_to_individuals_outcomes[i])
						data_items_to_print[-1][str(domain_of_possible_outcomes_sorted)]=[base_aggregates[i][io] for io in domain_of_possible_outcomes_sorted] 
						# marginal_infos
						# entities_outcomes
						
					for u in sorted(row['full_group_support']):
						data_groups_to_print.append(considered_users_1_sorted[id_individuals_to_indices[u]].copy())
						if u in all_individuals_to_entities_outcomes:
							for i in sorted (row['full_context_support']):
								if i in all_individuals_to_entities_outcomes[u]:
									data_groups_outcomes_to_print.append({entities_id_attribute:considered_items_sorted[id_entities_to_indices[i]][entities_id_attribute],individuals_id_attribute:considered_users_1_sorted[id_individuals_to_indices[u]][individuals_id_attribute],'outcome':all_individuals_to_entities_outcomes[u][i]})


					#description_attributes_objects,description_attributes_individuals,
					aggregate_items={k:'*' for k in process_outcome_dataset.STATS['items_header']}
					for k,pk in zip(description_attributes_objects,row['pattern_full'][1]):
						aggregate_items[k['name']]=pk
					aggregate_items['krippendorff_alpha'] = row['reliability_context']
					aggregate_items['nb_voters'] = sum([len(all_entities_to_individuals_outcomes[i]) for i in row['full_context_support']])
					aggregate_items[str(domain_of_possible_outcomes_sorted)]=[sum(record[str(domain_of_possible_outcomes_sorted)][ik] for record in data_items_to_print) for ik,io in enumerate(domain_of_possible_outcomes_sorted)]
					data_items_to_print.sort(key=lambda x:x['krippendorff_alpha'],reverse=True)
					data_items_to_print.append(aggregate_items)
					#reliability_context.append('context')
						#print considered_items_sorted[i],fast_reliability_with_bounds(marginal_infos,base_aggregates,{i},domain_of_possible_outcomes,distance_function=distance_function)
					
					#if True:
					for xxx in data_items_to_print:
						xxx['Group']=row['pattern_full'][0]
						


					writeCSVwithHeader(data_items_to_print,detailed_results_destination+'pattern_entities_'+str(row['id_pattern'])+'.csv',process_outcome_dataset.STATS['items_header']+['krippendorff_alpha']+['nb_voters']+[str(domain_of_possible_outcomes_sorted)],delimiter='\t')
					writeCSVwithHeader(data_groups_to_print,detailed_results_destination+'pattern_group_'+str(row['id_pattern'])+'.csv',process_outcome_dataset.STATS['users_header'],delimiter='\t')
					writeCSVwithHeader(data_groups_outcomes_to_print,detailed_results_destination+'pattern_group_outcomes_'+str(row['id_pattern'])+'.csv',[entities_id_attribute,individuals_id_attribute]+['outcome'],delimiter='\t')
					
					#print "########################################"
					#raw_input('....')

				###########
	if detailed_results_destination is not None:
		if not os.path.exists(detailed_results_destination):
			os.makedirs(detailed_results_destination)
		else:
			if False:
				shutil.rmtree(detailed_results_destination)
				os.makedirs(detailed_results_destination)
		for p in lists_of_patterns:
			#print detailed_results_destination
			full_group_support=sorted(all_individuals_to_entities_outcomes) if 'full_group_support' not in p else sorted (p['full_group_support'])
			full_context_support=p['full_context_support']
			

			
			if ADD_SIMILARITY_MATRIX:
				sim_dict_ref,sim_dict_context=compute_similarity_matrix(considered_users_1_sorted,individuals_id_attribute,all_entities_to_individuals_outcomes,all_individuals_to_entities_outcomes,full_group_support,full_context_support,distance_function=distance_function,domain_of_possible_outcomes=domain_of_possible_outcomes,marginal_infos=marginal_infos,key_to_print=key_to_print)




				sim_ref=transformMatricFromDictToList(sim_dict_ref)
				sim_context=transformMatricFromDictToList(sim_dict_context)
				#print sim_context
				from heatmap.heatmap import generateHeatMap
				cp_matrice_pattern=generateHeatMap(sim_context,detailed_results_destination+str(p['id_pattern']).zfill(4)+'_pattern_context.jpg',vmin=0.,vmax=1.,showvalues_text=False,only_heatmap=True,organize=True)
				#print cp_matrice_pattern
				innerMatrix,rower,header_matrix=getInnerMatrix(cp_matrice_pattern)
				#print rower
				rower=[rower[r] for r in sorted(rower)]
				header_matrix=[header_matrix[r] for r in sorted(header_matrix)]
				rower_inv={v:key for key,v in enumerate(rower)}
				header_inv={v:key for key,v in enumerate(header_matrix)}

				innerMatrix_ref,rower_ref,header_ref=getInnerMatrix(sim_ref)
				#print rower_ref
				rower_ref=[rower_ref[r] for r in sorted(rower_ref)]
				header_ref=[header_ref[r] for r in sorted(header_ref)]
				rower_ref_inv={v:key for key,v in enumerate(rower_ref)}
				header_ref_inv={v:key for key,v in enumerate(header_ref)}


				new_inner_matrix=[[innerMatrix_ref[rower_ref_inv[rowVal]][header_ref_inv[headVal]] for headVal in header_matrix] for rowVal in rower]
				matrice_ref_new=getCompleteMatrix(new_inner_matrix,{xx:yy for xx,yy in enumerate(rower)},{xx:yy for xx,yy in enumerate(header_matrix)})

				generateHeatMap(matrice_ref_new,detailed_results_destination+str(p['id_pattern']).zfill(4)+'_ref.jpg',vmin=0.,vmax=1.,showvalues_text=False,only_heatmap=True,organize=False)
			if ADD_SIMILARITY_MATRIX:
				
				RELIABILITY_DATA_REPRESENTATION=[]
				for id_ind in full_group_support:
					row_corresponding_to_the_individuals={'id_rater':id_ind}
					row_corresponding_to_the_individuals.update({id_ent:all_individuals_to_entities_outcomes[id_ind].get(id_ent,' ') for id_ent in full_context_support})
					RELIABILITY_DATA_REPRESENTATION.append(row_corresponding_to_the_individuals)
				writeCSVwithHeader(RELIABILITY_DATA_REPRESENTATION,detailed_results_destination+'pattern_'+str(p['id_pattern'])+'.csv',['id_rater']+sorted(full_context_support),delimiter='\t')

	
	return lists_of_patterns


def main_to_work_with():
	parser = argparse.ArgumentParser(description='DEPICT-XPs')
	replace_ids=True
	parser.add_argument('file', metavar='ConfigurationFile', type=str,  help='the input configuration file')
	parser.add_argument('-q','--qualitative',metavar='qualitative',nargs='*',help='execute a qualitative test')
	parser.add_argument('-p','--performance',metavar='performance',nargs='*',help='execute a perofrmance test')
	

	parser.add_argument('--first_run',action='store_true',help='first_run')
	parser.add_argument('--no_closure',action='store_true',help='use closure operator')
	parser.add_argument('--no_pruning',action='store_true',help='use pruning')
	parser.add_argument('--using_approxmations',action='store_true',help='use cdonfidence interval approximation rather than the empirical one')
	parser.add_argument('--nb_random_sample',metavar='nb_random_sample', nargs='?',help='the_number_of_sample_to_draw',type=int,default=5000,const=5000)
	

	parser.add_argument('--nb_attributes_individuals',metavar='nb_attributes_individuals', nargs='?',help='XXX',type=int,default=100000,const=100000)
	parser.add_argument('--nb_attributes_entities',metavar='nb_attributes_entities', nargs='?',help='XXX',type=int,default=100000,const=100000)
	

	parser.add_argument('--nb_items_individuals',metavar='nb_items_individuals', nargs='?',help='XXX',type=float,default=100000,const=100000)
	parser.add_argument('--nb_items_entities',metavar='nb_items_entities', nargs='?',help='XXX',type=float,default=100000,const=100000)
	
	parser.add_argument('--significance',metavar='significance', nargs='?',help='XXX',type=float,default=0.05,const=0.05)

	parser.add_argument('--threshold_objects',metavar='threshold_objects', nargs='?',help='XXX',type=float,default=0,const=0)
	parser.add_argument('--threshold_individuals',metavar='threshold_individuals', nargs='?',help='XXX',type=float,default=0,const=0)

	parser.add_argument('--compute_bootstrap_ci',action='store_true',help='compute_bootstrap_ci')
	parser.add_argument('--no_additional_files',action='store_true',help='no_additional_files')

	parser.add_argument('--results_destination',metavar='results_destination', nargs='?',help='XXX',type=str,default='',const='')
	
	
	parser.add_argument('--compute_empirical_distribution',action='store_true',help='compute_bootstrap_ci')
	parser.add_argument('--compute_p_value',action='store_true',help='compute_p_value')
	parser.add_argument('--no_concept_of_generality',action='store_true',help='no_concept_of_generality')
	#parser.add_argument('--similarity_matrix',action='store_true',help='similarity_matrix')
	parser.add_argument('--similarity_matrix',metavar='similarity_matrix',nargs='*',help='similarity_matrix',type=str)

	parser.add_argument('-f','--figure',metavar='figure',nargs='*',help='show a figure starting from a performance test')
	parser.add_argument('--do_not_plot_bars',action='store_true',help='do not plot bars')
	parser.add_argument('--do_not_plot_time',action='store_true',help='do not plot time')
	parser.add_argument('--fig_algos', metavar='fig_algos',nargs='*',help='select the algorithms to show in the figures',type=str)
	parser.add_argument('--x_axis_attribute',metavar='x_axis_attribute',type=str,help='which attribute to vary')
	parser.add_argument('--linear_scale_bars',action='store_true',help='Log linear_scale')
	parser.add_argument('--linear_scale_time',action='store_true',help='Log linear_scale')
	parser.add_argument('--taylorapprox',action='store_true',help='taylorapprox')
	args=parser.parse_args()

	if type(args.qualitative)  is list or type(args.performance)  is list:
		print 'Qualitative Experiments :) '
		json_file_path=args.file
		json_config=readJSON_stringifyUnicodes(json_file_path)
		PROFILING=False
		if PROFILING:
			pr = cProfile.Profile()
			pr.enable()


		json_config['closure_operator']=not args.no_closure
		json_config['pruning']=not args.no_pruning
		json_config['using_approxmations']=args.using_approxmations
		json_config['nb_random_sample']=args.nb_random_sample
		
		json_config['nb_items_individuals']=args.nb_items_individuals
		json_config['nb_items_entities']=args.nb_items_entities
		json_config['threshold_quality']=args.significance


		json_config['nb_attributes_individuals']=args.nb_attributes_individuals
		json_config['nb_attributes_entities']=args.nb_attributes_entities


		json_config['threshold_objects']=args.threshold_objects if args.threshold_objects>0 else json_config['threshold_objects']
		json_config['threshold_individuals']=args.threshold_individuals if args.threshold_individuals>0 else json_config['threshold_individuals']


		json_config['results_destination']=args.results_destination if len(args.results_destination)>0 else json_config['results_destination']
		json_config['compute_bootstrap_ci']=args.compute_bootstrap_ci if args.compute_bootstrap_ci else json_config.get("compute_bootstrap_ci",False) 
		json_config['compute_p_value']=args.compute_p_value
		json_config['no_concept_of_generality']=args.no_concept_of_generality
		json_config['similarity_matrix'] = type(args.similarity_matrix)  is list
		json_config['key_to_print'] = args.similarity_matrix if type(args.similarity_matrix)  is list else None
		# print json_config['nb_random_sample']
		# raw_input('...')


		lists_of_patterns=Depict_input_config(json_config,performance_test=(type(args.performance)  is list),compute_empirical_distribution=args.compute_empirical_distribution,replace_ids=replace_ids, first_run=args.first_run,no_additional_files=args.no_additional_files)
		if PROFILING:
			pr.disable()
			ps = pstats.Stats(pr)
			ps.sort_stats('cumulative').print_stats(30) #time
	elif type(args.figure) is list:	
		if args.taylorapprox:
			dataset_test,h= readCSVwithHeader(args.file,delimiter='\t',numberHeader=['error','error_empirical'],arrayHeader=['confidence_interval_taylor','confidence_interval_empirical','confidence_interval_empirical_percentile'])
			average_error=sum([row['error'] for row in dataset_test])/float(len(dataset_test))
			standard_deviation=(sum([(row['error'] - average_error)**2 for row in dataset_test])/(float(len(dataset_test))-1))**(0.5)

			average_error_emp=sum([row['error_empirical'] for row in dataset_test])/float(len(dataset_test))
			standard_deviation_emp=(sum([(row['error_empirical'] - average_error)**2 for row in dataset_test])/(float(len(dataset_test))-1))**(0.5)

			nb_larger=sum(1 if float(row['confidence_interval_taylor'][0])<=float(row['confidence_interval_empirical_percentile'][0]) and float(row['confidence_interval_taylor'][1])>=float(row['confidence_interval_empirical_percentile'][1]) else 0 for row in dataset_test)

			print average_error,standard_deviation,standard_deviation/(float(len(dataset_test))**(0.5)) 
			writeCSVwithHeader([{'filename' : args.file,'avg':average_error,'std':standard_deviation,'avgemp':average_error_emp,'stdemp':standard_deviation_emp,'nb_larger':nb_larger}],'./test_summary.csv',selectedHeader=['filename','avg','std','avgemp','stdemp','nb_larger'], flagWriteHeader=args.first_run)
		else:
			from plotter.perfPlotter import plotPerf,plotRW,plotHistograms_edf
			
			activated=args.fig_algos if args.fig_algos is not None and len(args.fig_algos)>0 else ["DEPICT"]
			json_file_path=args.file

			plotPerf(json_file_path,args.x_axis_attribute,show_legend=True,activated=activated,BAR_LOG_SCALE=not args.linear_scale_bars,TIME_LOG_SCALE=not args.linear_scale_time,plot_bars = not args.do_not_plot_bars, plot_time = not args.do_not_plot_time)

if __name__ == '__main__':
	main_to_work_with()

if __name__ == '__main2__':
	
	METHOD_TO_USE="CONFLICTUAL"#"CONSENSUAL"#"PVALUE"

	while False:

		V=[float(randint(3,155)) for _ in range(15)]
		W=[float(uniform(0.4,0.8)) for _ in range(15)]
		VW=[ (x*y,y) for x,y in zip(V,W)]
		example=[(1.1*1, 1), (0.9*10, 10)]
		VW=example
		newt_weird=newton_optimized_weird_correction(deepcopy(VW),1,False)
		exa_weird=exact_computing_weird_correction(deepcopy(VW),1,False)
		epst_weird=epstein_max_and_min_weird_correction(deepcopy(VW),1,False)
		newt=newton_optimized(deepcopy(VW),1,False)
		exa=exact_computing(deepcopy(VW),1,False)
		epst=epstein_max_and_min(deepcopy(VW),1,False)
		print newt,newt_weird
		print exa,exa_weird
		print epst,epst_weird
		raw_input('....')
		break
		#example=[(19650., 262.), (43615., 671.), (65512., 862.), (8484., 707.), (14520., 484.), (41904., 776.), (7326., 333.), (63744., 768.), (35332., 484.), (27945., 345.)]
		#VW=example



		#print VW
		newt=newton_optimized(deepcopy(VW),12,False)
		epst=epstein_max_and_min(deepcopy(VW),12,False)

		
		print newt
		print epst
		print epst/newt

		newt=newton_optimized(deepcopy(VW),12,True)
		epst=epstein_max_and_min(deepcopy(VW),12,True)

		print newt
		print epst
		raw_input('....')
		break
		# print exact_computing(VW,4)
		if newt!=epst:
			print newt
			print epst
			print newt-epst
			break
	# print newton_optimized(VW,12,True)
	# print epstein_max_and_min(VW,12,True)
	#print (VW)
	#print newton_optimized(VW,12,True)
	# VW2=[ (200*y - (x*y), (y)) for x,y in zip(V,W)]
	# print 200-random_epstein_exact(VW2,12)
	#raw_input('....')
	#main_1(False)
	

	#d,h=readCSVwithHeader("flickr_data.csv",delimiter=',')
	#print len(d)
	#print len({x[' title'] for x in d})
	#raw_input('....')
	numbers_header=["VOTE_DATE","GEOSIZE","POP16","POP16_PERCENTAGE","GDP_BILLIONS","NB_SEATS_IN_EU","EU_MEMBER_SINCE"]
	HMT_header=["PROCEDURE_SUBJECT","LANGUAGES"]

	entities_file_path=".\\Datasets\\EPD8DEDUP\\items.csv"
	individuals_file_path=".\\Datasets\\EPD8DEDUP\\users.csv"
	outcomes_file=".\\Datasets\\EPD8DEDUP\\reviews.csv"

	attributes_context=[]
	attributes_individuals=[]


	inidividuals_scope=[
		{'dimensionName':'COUNTRY','inSet':{'France'}},#,
		#{'dimensionName':'GROUPE_ID','inSet':{'S&D'}}
	]
	ratings_to_remove={'Abstain'}
	ratings_to_remove=set()
	entities_metadata,individuals_metadata,all_individuals_to_entities_outcomes,all_entities_to_individuals_outcomes,domain_of_possible_outcomes,outcomes_considered,entities_id_attribute,individuals_id_attribute,considered_items_sorted,considered_users_1_sorted,considered_users_2_sorted,nb_outcome_considered =\
	process_outcome_dataset(entities_file_path,individuals_file_path,outcomes_file,numeric_attrs=numbers_header,array_attrs=HMT_header,outcome_attrs=None,method_aggregation_outcome='SYMBOLIC_MAJORITY',
							itemsScope=[],users_1_Scope=inidividuals_scope,users_2_Scope=inidividuals_scope,ratings_to_remove=ratings_to_remove,nb_items=float('inf'),nb_individuals=float('inf'),attributes_to_consider=attributes_context+attributes_individuals,
							nb_items_entities=float('inf'),nb_items_individuals=float('inf'), hmt_to_itemset=False,
							delimiter='\t')




	DATASET_STATISTIC_COMPUTING=False
	if DATASET_STATISTIC_COMPUTING:
		description_attributes_items=[{'name':'PROCEDURE_SUBJECT', 'type':'themes'}]
		description_attributes_users=[{'name':'COUNTRY','type':'simple'}]
		items_id_attribute=entities_id_attribute
		users_id_attribute=individuals_id_attribute
		print 'nb_entities : ',len(considered_items_sorted)
		print 'nb_users : ',len(individuals_metadata)
		print 'nb_users_1 : ',len(considered_users_1_sorted)
		print 'nb_users_2 : ',len(considered_users_2_sorted)
		print 'nb_reviews : ',nb_outcome_considered
		print 'nb_attrs_entities : ', len(description_attributes_items)
		print 'nb_attrs_individuals : ', len(description_attributes_users)
		_,_,conf=next(enumerator_complex_cbo_init_new_config(considered_items_sorted,description_attributes_items))

		for ind_attr,attr in enumerate(conf['attributes']):
			if attr['type']=='themes':
				from enumerator.enumerator_attribute_themes2 import maximum_tree
				nb_tags_explicit_implicit_avg=0;nb_tags_explicit_implicit_max=0
				nb_tags_explicit_avg=0;nb_tags_explicit_avg_max=0
				

				for x in conf['allindex']:
					nb_tags_explicit_implicit_avg+= len(x[attr['name']])
					nb_tags_explicit_implicit_max=max(nb_tags_explicit_implicit_max,len(x[attr['name']]))
					nb_tags_explicit_avg+=len(maximum_tree(attr['domain'],x[attr['name']]))
					nb_tags_explicit_avg_max=max(nb_tags_explicit_avg_max,len(maximum_tree(attr['domain'],x[attr['name']])))
				nb_tags_explicit_implicit_avg/=float(len(conf['allindex']))
				nb_tags_explicit_avg/=float(len(conf['allindex']))
				print '\t', attr['name'], attr['type'], len(attr['domain']),'%.2f'%nb_tags_explicit_implicit_avg,'%.2f'%nb_tags_explicit_avg,nb_tags_explicit_implicit_max,nb_tags_explicit_avg_max

			else:
				print '\t', attr['name'], attr['type'], len(attr['domain'])

		all_avgs=0.
		nb_items_per_entities={}
		for ind,x in enumerate(conf['allindex']):
			avg_nb_items=0.
			for ind_attr,attr in enumerate(conf['attributes']):
				if attr['type']=='themes':
					avg_nb_items+=len(x[attr['name']])
				elif attr['type']=='simple':
					avg_nb_items+=1
				elif attr['type']=='numeric':
					avg_nb_items+=len(attr['domain'])+1
					#print (len(attr['domain'])-bisect_left(attr['domain'],x[attr['name']]))+(bisect_left(attr['domain'],x[attr['name']]))+1
			#print avg_nb_items
			nb_items_per_entities[considered_items_sorted[ind][items_id_attribute]]=avg_nb_items
			all_avgs+=avg_nb_items
		print 'nb_itemset : ', conf['nb_itemset'], all_avgs/len(conf['allindex'])
		nb_itemset_entities=conf['nb_itemset']

		_,_,conf=next(enumerator_complex_cbo_init_new_config(considered_users_1_sorted,description_attributes_users))
		for ind_attr,attr in enumerate(conf['attributes']):
			if attr['type']=='themes':
				from enumerator.enumerator_attribute_themes2 import maximum_tree
				nb_tags_explicit_implicit_avg=0;nb_tags_explicit_implicit_max=0
				nb_tags_explicit_avg=0;nb_tags_explicit_avg_max=0
				for x in conf['allindex']:
					nb_tags_explicit_implicit_avg+= len(x[attr['name']])
					nb_tags_explicit_implicit_max=max(nb_tags_explicit_implicit_max,len(x[attr['name']]))
					nb_tags_explicit_avg+=len(maximum_tree(attr['domain'],x[attr['name']]))
					nb_tags_explicit_avg_max=max(nb_tags_explicit_avg_max,len(maximum_tree(attr['domain'],x[attr['name']])))
				nb_tags_explicit_implicit_avg/=float(len(conf['allindex']))
				nb_tags_explicit_avg/=float(len(conf['allindex']))
				print '\t', attr['name'], attr['type'], len(attr['domain']),'%.2f'%nb_tags_explicit_implicit_avg,'%.2f'%nb_tags_explicit_avg,nb_tags_explicit_implicit_max,nb_tags_explicit_avg_max

			else:
				print '\t', attr['name'], attr['type'], len(attr['domain'])

		all_avgs=0.
		nb_items_per_individual={}
		for ind,x in enumerate(conf['allindex']):
			avg_nb_items=0.
			for ind_attr,attr in enumerate(conf['attributes']):
				if attr['type']=='themes':
					avg_nb_items+=len(x[attr['name']])
				elif attr['type']=='simple':
					avg_nb_items+=1
				elif attr['type']=='numeric':
					avg_nb_items+=len(attr['domain'])+1
			nb_items_per_individual[considered_users_1_sorted[ind][users_id_attribute]]=avg_nb_items

			#print (len(attr['domain'])-bisect_left(attr['domain'],x[attr['name']]))+(bisect_left(attr['domain'],x[attr['name']]))+1
			#print avg_nb_items
			all_avgs+=avg_nb_items

		print 'nb_itemset : ', conf['nb_itemset'], all_avgs/len(conf['allindex'])
		nb_itemset_individuals=conf['nb_itemset']
		nb_itemset_individuals_avg=all_avgs/len(conf['allindex'])
		entities_to_users_outcomes={}
		for u in all_individuals_to_entities_outcomes:
			for e in all_individuals_to_entities_outcomes[u]:
				if e not in entities_to_users_outcomes:
					entities_to_users_outcomes[e]=set()
				entities_to_users_outcomes[e]|={u}

		nb_e_u_s=sum(len(x) for x in entities_to_users_outcomes.values())
		nb_e_uu_s=sum(len(x)*len(x) for x in entities_to_users_outcomes.values())
		#nb_avg_itemset=sum(len(entities_to_users_outcomes[x])*len(entities_to_users_outcomes[x])*(nb_items_per_entities[x]+nb_itemset_individuals_avg) for x in entities_to_users_outcomes)/float(nb_e_uu_s)
		nb_avg_itemset=sum((nb_items_per_entities[x]+nb_items_per_individual[y])*len(entities_to_users_outcomes[x]) for x in entities_to_users_outcomes for y in entities_to_users_outcomes[x])/float(nb_e_uu_s)
		print 'nb_considered_reviews ', nb_e_u_s
		print 'nb_considered_entries_cartesian ', nb_e_uu_s
		print 'nb_itemsets_full ', nb_itemset_entities+nb_itemset_individuals*2
		print 'nb_itemsets_avg_per_transaction ', nb_avg_itemset
		raw_input('...')


	# print len(considered_items_sorted)
	# print len(set(x[entities_id_attribute] for x in considered_items_sorted))
	# print 'DONE PREPROCESSING'
	# raw_input('****')
	
	
	# print all_entities_to_individuals_outcomes[all_entities_to_individuals_outcomes.keys()[0]]
	# print domain_of_possible_outcomes
	
	
	set_of_ballots=all_entities_to_individuals_outcomes.viewkeys()

	domain_of_possible_outcomes=sorted(domain_of_possible_outcomes)
	print len(set_of_ballots)
	print len(all_individuals_to_entities_outcomes)
	print len({i for i in all_individuals_to_entities_outcomes if len(all_individuals_to_entities_outcomes[i])>0})
	nb_raters=len({i for i in all_individuals_to_entities_outcomes if len(all_individuals_to_entities_outcomes[i])>0})
	print nb_outcome_considered
	#raw_input('....')
	#raw_input('NO BITCH ! ')
	base_aggregates,marginal_infos=pre_compute_base_aggregates_for_kripendorff(all_entities_to_individuals_outcomes,domain_of_possible_outcomes,all_individuals_to_entities_outcomes)
	# print base_aggregates[all_entities_to_individuals_outcomes.keys()[0]]
	# print marginal_infos

	if False:
		colors=['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499', '#117733']
		color_index=0
		for k in range(50,201,+25):
			s=set(range(k))
			a,b,edf=reliability_CI_WITHOUT_REPLACEMENT_DISTRIBUTION(marginal_infos,base_aggregates,s,domain_of_possible_outcomes,set_of_ballots,distance_function=distance_nominal,error=0.01,nb_per_bag=1000,nb_bootstraps=5)
			c,l,largeur,e = histogram(edf, 50)
			c=[t/sum(c) for t in c]
			xaxis_edf=[l+largeur * i for i in range(50)]
			plt.plot(a,b,label='Line '+str(k),color=colors[color_index])
			plt.bar(xaxis_edf,c,largeur, label='Hist '+str(k),alpha=0.6,color=colors[color_index])
			color_index+=1
		plt.legend()	
		plt.show()
		raw_input('...')
		#raw_input('...')
	
	print 'reliability_full :  ',reliability(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal) , reliability_as_kilem_2(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal)
	#raw_input('STOP HERE')
	#print reliability_CI_KILEM_GWET_BLB(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,set_of_ballots,distance_function=distance_nominal)
	
	#raw_input('...')
	print 'Start'
	st=time()
	#lists_of_patterns=find_exeptional_contexts(considered_items_sorted,[{'name':'PROCEDURE_SUBJECT', 'type':'themes'}],marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes)
	if False:
		PROFILING=True
		if PROFILING:
			pr = cProfile.Profile()
			pr.enable()
		lists_of_patterns=find_exeptional_contexts(considered_items_sorted,[{'name':'PROCEDURE_SUBJECT', 'type':'themes'},{'name':'VOTE_DATE', 'type':'numeric'}],marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes)
		if PROFILING:
			pr.disable()
			ps = pstats.Stats(pr)
			ps.sort_stats('time').print_stats(20) #cumulative
			raw_input('.....')

		print 'NB Exceptional patterns found : ',len(lists_of_patterns)
		print 'timespent : ', time()-st
		for p in lists_of_patterns:
			print p
			#reliability(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal) 
			raw_input('...')

	st=time()
	


	PROFILING=False
	if PROFILING:
		pr = cProfile.Profile()
		pr.enable()
	
	if METHOD_TO_USE=="CONFLICTUAL":
		print "TOP K CONFLICTUAL ...." 
		quality_threshold=0.00000001
		support_threshold=10
		lists_of_patterns=find_exeptional_conflictual_contexts(considered_items_sorted,[{'name':'PROCEDURE_SUBJECT', 'type':'themes'}],entities_id_attribute,marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,support_threshold=support_threshold,quality_threshold=quality_threshold)
	elif  METHOD_TO_USE=="PVALUE":
		lists_of_patterns=find_exeptional_contexts_with_CIs(considered_items_sorted,[{'name':'PROCEDURE_SUBJECT', 'type':'themes'}],entities_id_attribute,marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes)



	print 'NB Exceptional patterns found : ',len(lists_of_patterns)
	print 'timespent : ', time()-st
	if PROFILING:
		pr.disable()
		ps = pstats.Stats(pr)
		ps.sort_stats('time').print_stats(20) #cumulative
		#raw_input('.....')
	if True:
		for p in lists_of_patterns:
			print p[0],p[2],sum([len(all_entities_to_individuals_outcomes[x]) for x in p[5]]),p[1],p[4],p[3]
		#raw_input('....')
	#raw_input('*********************************')
	reliability_full=reliability(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal)
	l_div=lists_of_patterns
	#l_div=get_top_k_div_from_a_pattern_set(lists_of_patterns,threshold_sim=1.2,k=1000)

	l_div=get_top_k_div_from_a_pattern_set_cumulative_quality(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,lists_of_patterns,distance_function=distance_nominal,k=1000)

	print len(l_div)
	results=[]
	for p in l_div:
		results.append(
			{
				'context':p[0],
				'support':p[2],
				'nbOutcomes':sum([len(all_entities_to_individuals_outcomes[x]) for x in p[5]]),
				'ContextAgreement':'%.2f' % p[1],
				'ReferenceAgreement':'%.2f' % reliability_full,
				'RelatedConfidenceInterval':[float('%.2f' % p[4][0]),float('%.2f' % p[4][1])],
				'Situation':'Conflictual' if p[1] < reliability_full else 'Consensual'   
			}
		)
		if False:
			print p[0],p[2],sum([len(all_entities_to_individuals_outcomes[x]) for x in p[5]]),p[1],p[4],p[3]
		#raw_input('....')
	writeCSVwithHeader(results,'./results.csv',['context','support' ,'nbOutcomes' , 'ReferenceAgreement','ContextAgreement', 'RelatedConfidenceInterval','Situation'],delimiter='\t')
	#######################################################################
	if False:
		PROFILING=False
		if PROFILING:
			pr = cProfile.Profile()
			pr.enable()
		print reliability_CI_KILEM_GWET_BLB(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,set_of_ballots,distance_function=distance_nominal)
		raw_input('...')
		if PROFILING:
			pr.disable()
			ps = pstats.Stats(pr)
			ps.sort_stats('time').print_stats(20) #cumulative
			raw_input('.....')

		reliability_full=reliability(marginal_infos,base_aggregates,set_of_ballots,domain_of_possible_outcomes,distance_function=distance_nominal)
		print reliability_full
		raw_input('...')


		enumerator_contexts=enumerator_complex_cbo_init_new_config(considered_items_sorted, [{'name':'PROCEDURE_SUBJECT', 'type':'themes'}],threshold=15)
		lists_of_patterns=[]
		for e_p,e_label,e_config in enumerator_contexts:
			context_pat=pattern_printer(e_label,['themes'])
			entities_context=set(x[entities_id_attribute] for x in e_config['support'])
			reliability_pattern=reliability(marginal_infos,base_aggregates,entities_context,domain_of_possible_outcomes,distance_function=distance_nominal)
			print 'CI : ',reliability_CI_KILEM_GWET_BLB(marginal_infos,base_aggregates,entities_context,domain_of_possible_outcomes,set_of_ballots,distance_function=distance_nominal),reliability_pattern
			raw_input('....')
			lists_of_patterns.append((context_pat,reliability_pattern,len(entities_context),abs(reliability_pattern-reliability_full)))
		lists_of_patterns=sorted(lists_of_patterns,key=lambda x:x[-1],reverse=True)
		for p in lists_of_patterns:
			print p
			raw_input('....')
	#######################################################################
	# print considered_items_sorted[	# print considered_users_1_sorted


	#find_exceptional_intra_group_agreement_patterns()

