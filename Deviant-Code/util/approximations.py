from itertools import product, combinations
from numpy.random import choice,multinomial,shuffle,geometric
from random import random, randint
import scipy.stats as st
from scipy.misc import comb
from math import floor,ceil
from bisect import bisect_left

from plotter.perfPlotter import plotHistograms_to_test_edf

import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


def avg(l):
	return (1/float(len(l)))*sum(l)

def stddev(l):
	avg_l=avg(l)
	
	return (sum((x-avg_l)**2 for x in l)/float(len(l)-1))**0.5

def normalize_distribution(l):
	avg_l,std_l=avg(l),stddev(l)
	return [(x-avg_l)/std_l for x in l]


def prediction_interval(l,alpha=0.05):
	talpha_up=st.t.ppf((1-alpha/2.),len(l)-1)
	avg_l,std_l=avg(l),stddev(l)
	return [avg_l-std_l*talpha_up* (1+1/float(len(l)))**(0.5) , avg_l+std_l*talpha_up* (1+1/float(len(l)))**(0.5) ]  

def all_constants(V,W):
	n=float(len(V))
	u_v=avg(V)
	u_v2=avg([x**2 for x in V])
	
	u_vv=(n/(n-1))*(u_v*u_v-(1/n)*(u_v2))
	#u_vv=((1/(n*(n-1)))* sum([V[i1] * V[i2] for (i1,i2) in product(range(int(n)),range(int(n))) if i1!=i2 ]))

	u_w=avg(W)
	u_w2=avg([x**2 for x in W])
	
	u_ww=(n/(n-1))*(u_w*u_w-(1/n)*(u_w2))
	#u_ww=((1/(n*(n-1)))* sum([W[i1] * W[i2] for (i1,i2) in product(range(int(n)),range(int(n))) if i1!=i2 ]))

	u_z=avg([x*y for x,y in zip(V,W)])
	u_vw=(n/(n-1))*(u_v*u_w-(1/n)*(u_z))
	
	#u_vw=((1/(n*(n-1)))* sum([V[i1] * W[i2] for (i1,i2) in product(range(int(n)),range(int(n))) if i1!=i2 ]))

	return n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw



def beta_w(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw):
	#n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(X,Y)
	return (u_w2 - u_ww)/(u_w*u_w) - (u_z - u_vw)/(u_w*u_v)

def beta_v(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw):
	#n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(X,Y)
	return (u_v2 - u_vv)/(u_v*u_v) - (u_z - u_vw)/(u_w*u_v)



def expectation_ratio_approx(V,W,k):
	k=float(k)

	n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V,W)
	B_w= beta_w(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)

	return (1/k) * (u_v/u_w) * B_w + (u_v/u_w) * (u_ww/(u_w*u_w) - u_vw/(u_v*u_w) + 1)

def expectation_ratio_approx_firstOrder(V,W,k):
	#k=float(k)

	n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V,W)
	#B_w= beta_w(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)

	return u_v/u_w


def variance_ratio_approx(V,W,k):
	k=float(k)
	n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V,W)
	B_w= beta_w(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
	B_v= beta_v(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)



	cov_vw=(1/k)*(u_z-u_vw) + u_vw- u_v*u_w
	var_v=(1/k)*(u_v2-u_vv) + u_vv- u_v*u_v
	var_w=(1/k)*(u_w2-u_ww) + u_ww- u_w*u_w

	return abs((1/k)* ((u_v**2/u_w**2)) * (B_w+B_v) + ((u_v**2/u_w**2))* ((u_ww/(u_w*u_w)) + (u_vv/(u_v*u_v)) -  2 * (u_vw/(u_v*u_w))))
	

	#return abs(var_v/(u_w*u_w)-2*(u_v/(u_w**3))*cov_vw+ (u_v**2/u_w**4)*var_w)





def expectation_ratio_real(V,W,k):
	nb=0.
	avg_total=0.
	for indices in combinations(range(len(V)),k):
		#print nb
		avg_now=sum(V[i] for i in indices)/float(sum(W[i] for i in indices))
		nb+=1
		avg_total+=avg_now

	return avg_total/nb

def variance_ratio_real(V,W,k):
	nb=0.
	var_total=0.
	avg_all_cmb=expectation_ratio_real(V,W,k)

	for indices in combinations(range(len(V)),k):
		var_now=(sum(V[i] for i in indices)/float(sum(W[i] for i in indices)) - avg_all_cmb)**2
		nb+=1
		var_total+=var_now

	return var_total/nb



def CI_approx(V,W,k,alpha,disagreement_expected,one_tail_left=False,one_tail_right=False):
	std=variance_ratio_approx(V,W,k)**(1/2.)
	#ste=std/(k**1/2.)
	exp=expectation_ratio_approx(V,W,k)
	n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V,W)
	B_w= beta_w(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
	B_v= beta_v(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
	# print ((u_v**2/u_w**2)) * (B_w+B_v),abs((u_v/u_w) * B_w) # One way is to compare for all k ? or at least compare 
	# print std,exp,((u_v**2/u_w**2))* ((u_ww/(u_w*u_w)) + (u_vv/(u_v*u_v)) -  2 * (u_vw/(u_v*u_w)))
	# print ((B_v+B_w)/(B_v**2))**(1/2.)

	# print (((B_v+B_w)/(B_v**2))**(1/2.) * (2-2**(1/2.)))**(-1.) , st.norm.cdf((((B_v+B_w)/(B_v**2))**(1/2.) * (2-2**(1/2.)))**(-1.))

	#raw_input('...')


	# print exp,std
	# raw_input('000000000000000')

	if one_tail_left:
		#return [exp-st.norm.ppf((1-alpha))*std, float('+inf')]
		return [1-(exp+st.norm.ppf((1-alpha))*std)/disagreement_expected, float('+inf')]
	if one_tail_right:
		return [float('-inf'), 1-(exp-st.norm.ppf((1-alpha))*std)/disagreement_expected]
	else:
		return [1-(exp+st.norm.ppf((1-alpha/2.))*std)/disagreement_expected,1-(exp-st.norm.ppf((1-alpha/2.))*std)/disagreement_expected]
		#return [exp-st.norm.ppf((1-alpha))*std,exp+st.norm.ppf((1-alpha))*std]
	#critical_value_right = st.norm.ppf(1-(alpha)) if one_tail_right else float('+inf')
	


def beta_w_2(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw):
	return (1/(n-1)) * ( (u_w2/(u_w**2)) - (u_z)/(u_v*u_w) )

def beta_v_2(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw):
	return (1/(n-1)) * ( (u_v2/(u_v**2)) - (u_z)/(u_v*u_w) )

def minimum_k_valid_approx(V,W,alpha):
	n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V,W)
	B_w= beta_w_2(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
	B_v= beta_v_2(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
	coeff_alpha=st.norm.ppf((1-alpha/2.))

	return n/ ( ((coeff_alpha**2) *(B_w+B_v)/(4*(B_w**2)))+1)

def CI_approx_second(V,W,k,alpha,disagreement_expected,one_tail_left=False,one_tail_right=False,constants_already_computed=False,use_student_instrad=True):
	k=float(k)
	
	if not constants_already_computed:
		
		n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V,W)
		B_w= beta_w_2(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
		B_v= beta_v_2(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
		CI_approx_second.constants=(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw,B_v,B_w)
	else:
		n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw,B_v,B_w=CI_approx_second.constants
	exp= (n/k-1)* (u_v/u_w)*B_w + (u_v/u_w)
	var= (n/k-1)* (u_v**2/u_w**2)*(B_w + B_v)# + (u_v/u_w)
	std=var**(1/2.)



	if one_tail_left:
		#return [exp-st.norm.ppf((1-alpha))*std, float('+inf')]
		return [1-(exp+st.norm.ppf((1-alpha))*std)/disagreement_expected, float('+inf')]
	if one_tail_right:
		return [float('-inf'), 1-(exp-st.norm.ppf((1-alpha))*std)/disagreement_expected]
	else:
		return [1-(exp+st.norm.ppf((1-alpha/2.))*std)/disagreement_expected,1-(exp-st.norm.ppf((1-alpha/2.))*std)/disagreement_expected]


def CI_approx_second_with_p_value(V,W,k,alpha,disagreement_expected,one_tail_left=False,one_tail_right=False,constants_already_computed=False,use_student_instrad=True):
	k=float(k)
	if not constants_already_computed:
		n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V,W)
		B_w= beta_w_2(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
		B_v= beta_v_2(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
		CI_approx_second_with_p_value.constants=(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw,B_v,B_w)
	else:
		n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw,B_v,B_w=CI_approx_second_with_p_value.constants

	exp= (n/k-1)* (u_v/u_w)*B_w + (u_v/u_w)
	var= (n/k-1)* (u_v**2/u_w**2)*(B_w + B_v)# + (u_v/u_w)
	std=var**(1/2.)

	if one_tail_left:
		#return [exp-st.norm.ppf((1-alpha))*std, float('+inf')]
		return exp,std,[1-(exp+st.norm.ppf((1-alpha))*std)/disagreement_expected, float('+inf')]
	if one_tail_right:
		return exp,std,[float('-inf'), 1-(exp-st.norm.ppf((1-alpha))*std)/disagreement_expected]
	else:
		return 1-exp/disagreement_expected,std/disagreement_expected,[1-(exp+st.norm.ppf((1-alpha/2.))*std)/disagreement_expected,1-(exp-st.norm.ppf((1-alpha/2.))*std)/disagreement_expected]
		#stats.norm.cdf(x, mean, sigma)
def get_sample(arr, n_iter=None, sample_size=10, fast=True):
	n=len(arr)
	if fast:
		start_idx = (n_iter * sample_size) % n
		if start_idx + sample_size >= n:
			shuffle(arr)

		return arr[start_idx:start_idx+sample_size] 
	else:
		return choice(arr, sample_size, replace=False)

def collect_samples(arr,sample_size,n_samples,fast=False):
	shuffle(arr)
	for sample_n in xrange(n_samples):
		yield  get_sample(arr, n_iter=sample_n,sample_size=sample_size,fast=fast)


def CI_approx_empirical_comparison(V,W,alpha,disagreement_expected,nb_random_sample=1000,one_tail_left=False,one_tail_right=False,file_to_write_in=None,return_distribution=False,number_of_distributions=400):
	

	V_new=[w-v/disagreement_expected for w,v in zip(W,V)]

	distributions_to_return=[]
	dataset_to_write=[]

	# if True:
	# 	n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V,W)
	# 	B_w= beta_w(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
	# 	B_v= beta_v(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)

	# 	n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V_new,W)
	# 	B_w= beta_w(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
	# 	B_v_new= beta_v(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)

	# 	print B_w,B_v_new,B_w,minimum_k_valid_approx(V,W,alpha),minimum_k_valid_approx(V_new,W,alpha)
	# 	raw_input('....')
	n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V,W)
	B_w= beta_w(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)
	B_v= beta_v(n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw)

	coeff=st.norm.ppf((1-alpha/2.))

	func_on_k = lambda x : (x+1)**(x**(1/2.)) - (x)*(x+1)**(1/2.)

	func_get_conf = lambda x : func_on_k(x) *  coeff*((B_v+B_w)**1/2.)/abs(B_v)
	k_selected=1
	# for k in range(1,100):
	# 	print k,func_get_conf(k)
	# 	if func_get_conf(k)>=1:
	# 		k_selected=k
	# 		break
	# print '----------------------------------------------'
	min_thres_real=minimum_k_valid_approx(V,W,alpha)
	k_selected=int(ceil(minimum_k_valid_approx(V,W,alpha)))
	print 'MINIMUM K : ', k_selected,minimum_k_valid_approx(V,W,alpha),' nb Samples : ',nb_random_sample, "alpha : ", alpha, "C alpha",minimum_k_valid_approx(V,W,alpha)

	#raw_input('*')
	#for k in range(int(n)/2,k_selected,-50):
	#for k in range(int(n)-50,k_selected,-50):
	
	#[1000* x/10. for x in range(10)]
	NB_VALUE_TO_TEST_AGAINST= number_of_distributions#int(n/20.)
	NUMBER_OF_SETS_TO_TEST = [int((x+1) * n/float(NB_VALUE_TO_TEST_AGAINST)) for x in range(NB_VALUE_TO_TEST_AGAINST-1)]
	NUMBER_OF_SETS_TO_TEST=NUMBER_OF_SETS_TO_TEST[bisect_left(NUMBER_OF_SETS_TO_TEST,int(ceil((n*0.01)))):]
	print 'NB TESTS : ', len(NUMBER_OF_SETS_TO_TEST),' n = ', n 
	for k in NUMBER_OF_SETS_TO_TEST:
	#for k in range(int(n*(0.05)),int(n)-1):
		#CI=CI_approx(V,W,k,alpha,disagreement_expected)
		CI=CI_approx_second(V,W,k,alpha,disagreement_expected)
		#print k,'\t\t','%.5f'%CI[0],'\t','%.5f'%CI[1]
		#print k,'\t\t','%.5f'%CI_2[0],'\t','%.5f'%CI_2[1]
		#raw_input('****')
		nb_samples=nb_random_sample
		sampulas=[1.-(sum([V[i] for i in x])/sum([W[i] for i in x]))/disagreement_expected for x in collect_samples(range(len(V)),k,nb_samples,fast=False)]
		
		if return_distribution:
			distributions_to_return.append(sampulas)
		#p_value_of_the_test=st.normaltest(normalize_distribution(sampulas))#st.kstest(sampulas,'norm',N=100)#st.shapiro(sampulas)[1]
		if False:
			p_value_of_the_test=st.shapiro(normalize_distribution(sampulas)[:5000])
			p_value_of_the_test_2=st.normaltest(normalize_distribution(sampulas))
			print 'p - value - of the shapiro test : ', p_value_of_the_test[1], 'NORMALLY DISTRIBUTED ' if p_value_of_the_test[1]>=0.01 else 'NOT NORMALLY DISTRIBUTED '
			#raw_input('....')
			plotHistograms_to_test_edf(normalize_distribution(sampulas))
		l = sorted(sampulas)
		
		CI_EMP=l[int(len(l)*(alpha/2.))],l[int(len(l)*(1-alpha/2.))]
		if True:
			# talpha_up=st.t.ppf((1-alpha/2.),len(l)-1)
			# talpha_low=st.t.ppf((alpha/2.),len(l)-1)
			# avg_l,std_l=avg(l),stddev(l)
			CI_EMP= prediction_interval(sampulas,alpha)
			pval=max(st.shapiro(normalize_distribution(sampulas)[:5000])[1],st.normaltest(normalize_distribution(sampulas))[1],st.kstest(normalize_distribution(sampulas),'norm',N=1000)[1])
			#print pval,st.kstest(normalize_distribution(sampulas),'norm',N=1000)[1]
			#raw_input('...')
		CI_EMP_PERCENTILE=l[int(len(l)*(alpha/2.))],l[int(len(l)*(1-alpha/2.))]
		#CI_EMP=st.t.interval(alpha, len(l)-1, loc=mean, scale=1)

		dataset_to_write.append({
			'dataset_size':int(n),
			'alpha':alpha,
			'sample_size':int(k),
			'minimum_threshold_alpha':min_thres_real,#,k_selected,
			'confidence_interval_taylor':CI,
			'confidence_interval_empirical':CI_EMP,
			'confidence_interval_empirical_percentile':CI_EMP_PERCENTILE,
			'error': 1-abs(min(CI[1],CI_EMP[1])-max(CI[0],CI_EMP[0]))/abs(max(CI[1],CI_EMP[1])-min(CI[0],CI_EMP[0])),
			'error_empirical': 1-abs(min(CI[1],CI_EMP_PERCENTILE[1])-max(CI[0],CI_EMP_PERCENTILE[0]))/abs(max(CI[1],CI_EMP_PERCENTILE[1])-min(CI[0],CI_EMP_PERCENTILE[0])),
			'normal_distribution_pvalue': pval,
			'follows_normal': int(pval>=0.01)

		})

		if return_distribution:
			dataset_to_write[-1]["empirical_distribution"]=sampulas
			dataset_to_write[-1]["empirical_distribution_normalized"]=normalize_distribution(sampulas)

		yield dataset_to_write[-1]
		print k,'\t\t','%.5f'%CI[0],'\t','%.5f'%CI[1]
		print k,'\t\t','%.5f'%CI_EMP[0],'\t','%.5f'%CI_EMP[1] 
		print k,'\t\t','%.5f'%CI_EMP_PERCENTILE[0],'\t','%.5f'%CI_EMP_PERCENTILE[1] 
		print dataset_to_write[-1]['error'] , 1-abs(min(CI[1],CI_EMP_PERCENTILE[1])-max(CI[0],CI_EMP_PERCENTILE[0]))/abs(max(CI[1],CI_EMP_PERCENTILE[1])-min(CI[0],CI_EMP_PERCENTILE[0]))
		print '       '
		#raw_input('..........')
	#print '----------------------------------------------'
	#return dataset_to_write



def CI_real(V,W,k,alpha,one_tail_left=False,one_tail_right=False):
	std=variance_ratio_real(V,W,k)**(1/2.)
	exp=expectation_ratio_real(V,W,k)
	

	if one_tail_left:
		return [exp-st.norm.ppf((1-alpha))*std, float('+inf')]
	if one_tail_right:
		return [float('-inf'), exp+st.norm.ppf((1-alpha))*std]
	else:
		return [exp-st.norm.ppf((1-alpha/2.))*std,exp+st.norm.ppf((1-alpha/2.))*std]



if __name__ == '__main__':
	V=[randint(1,10) * random() for _ in range(30)]
	W=[randint(1,10) * random() for _ in range(30)]
	k=26
	print comb(30,k)
	raw_input(('...'))
	print expectation_ratio_real(V,W,k),variance_ratio_real(V,W,k)**(1/2.),expectation_ratio_approx(V,W,k),variance_ratio_approx(V,W,k)**(1/2.),