import math
from itertools import product, combinations
from numpy.random import choice,multinomial
from random import random, randint

def avg(S):
	return sum(S)/float(len(S))

def avg_all_cmb(G,k):
	nb=0.
	avg_total=0.
	for indices in combinations(range(len(G)),k):
		avg_now=sum(G[i] for i in indices)/float(k)
		nb+=1
		avg_total+=avg_now
	return avg_total/nb



def weighted_avg_all_cmb(G,W,k):
	nb=0.
	avg_total=0.
	for indices in combinations(range(len(G)),k):
		avg_now=sum(W[i]*G[i] for i in indices)/float(sum(W[i] for i in indices))
		nb+=1
		avg_total+=avg_now
	return avg_total/nb

def std_all_cmb_1(G,k):
	nb=0.
	avg_total=0.
	std_total=0.
	for indices in combinations(range(len(G)),k):
		std_total+=(avg([G[i] for i in indices])-avg(G))**2
		nb+=1
		
	return ((std_total)/nb)**(1/2.)

def std_all_cmb_5(G,k):
	n=float(len(G))
	cte=(sum(G[x]*G[y] if x!=y else 0 for x,y in product(range(len(G)),range(len(G)))))/(n*(n-1))
	total=(1/float(k))*(avg([x**2 for x in G])-cte)+cte
	total=total-avg(G)**2
	return total**(1/2.)

def samples_and_test(G,k,n=1000):
	edf=[]
	for _ in xrange(n):
		selected_batch_for_now=[G[x] for x in choice(range(len(G)), k, replace=False).tolist()]
		statistic=avg(selected_batch_for_now)
		edf.append(statistic)
	return ((1/float(n))*sum((x-avg(G))**2 for x in edf))**(1/2.)




def cov(l1,l2,k):
	n=float(len(l1))
	avg_1=avg(l1)
	avg_2=avg(l2)
	l12=[x*y for x,y in zip(l1,l2)]
	avg_12=avg(l12)
	cte_2=(1/(n*(n-1)))* sum([l1[i1] * l2[i2] for (i1,i2) in product(range(int(n)),range(int(n))) if i1!=i2 ])

	return (1/float(k)) * (avg_12 - cte_2) +  cte_2 - avg_1*avg_2

def variance_sample_proportion(G,k):
	n=float(len(G))
	
	cte=(sum(G[x]*G[y] if x!=y else 0 for x,y in product(range(len(G)),range(len(G)))))/(n*(n-1))
	total=(1/float(k))*(avg([x**2 for x in G])-cte)+cte
	total=total-avg(G)**2
	return total


def expectation_ratio(l1,l2,k):
	return (avg(l1)/avg(l2)) + variance_sample_proportion(l2,k)*(avg(l1)/(avg(l2)**3)) - (cov(l1,l2,k)/(avg(l2)**2))


def all_constants(V,W):
	n=len(V)
	u_v=avg(V)
	u_v2=avg([x**2 for x in V])
	u_vv=((1/(n*(n-1)))* sum([V[i1] * V[i2] for (i1,i2) in product(range(int(n)),range(int(n))) if i1!=i2 ]))

	u_w=avg(W)
	u_w2=avg([x**2 for x in W])
	u_ww=((1/(n*(n-1)))* sum([W[i1] * W[i2] for (i1,i2) in product(range(int(n)),range(int(n))) if i1!=i2 ]))

	u_z=avg([x*y for x,y in zip(V,W)])
	u_vw=((1/(n*(n-1)))* sum([V[i1] * W[i2] for (i1,i2) in product(range(int(n)),range(int(n))) if i1!=i2 ]))

	return n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw

def expectation_ratio_approx_2(V,W,k):
	k=float(k)
	n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V,W)
	return (1/k)* ( (u_v/u_w)*(   ((u_w2-u_ww)/(u_w*u_w)) - ((u_z-u_vw)/(u_v*u_w)) )) + ((u_v/u_w) * ( (u_ww/(u_w*u_w)) - (u_vw/(u_v*u_w)) +1  ))



def variance_ratio_approx_2(V,W,k):
	k=float(k)
	n,u_v,u_v2,u_vv,u_w,u_w2,u_ww,u_z,u_vw=all_constants(V,W)
		
	cov_vw=(1/k)*(u_z-u_vw) + u_vw- u_v*u_w
	var_v=(1/k)*(u_v2-u_vv) + u_vv- u_v*u_v
	var_w=(1/k)*(u_w2-u_ww) + u_ww- u_w*u_w


	return var_v/(u_w*u_w)-2*(u_v/(u_w**3))*cov_vw+ (u_v**2/u_w**4)*var_w

	#return (1/k)* ( (u_v/u_w)*(   ((u_w2-u_ww)/(u_w*u_w)) - ((u_z-u_vw)/(u_v*u_w)) )) + ( (u_v/u_w) * ( (u_ww/(u_w*u_w)) - (u_vw/(u_v*u_w)) +1  ))


def variance_ratio_sampling(V,W,k,nb_try=10000):
	nb=0.
	var_total=0.
	list_of_unities=range(len(V))
	avg_all_cmb=expectation_ratio_sampling(V,W,k,nb_try=10000)
	for _ in range(nb_try):
		indices=choice(list_of_unities, k, replace=False).tolist()
		var_now=(sum(V[i] for i in indices)/float(sum(W[i] for i in indices)) - avg_all_cmb)**2
		nb+=1
		var_total+=var_now

	return var_total/nb




G=[random() for _ in xrange(50)]
W=[random() for k in range(50)]
k=10



