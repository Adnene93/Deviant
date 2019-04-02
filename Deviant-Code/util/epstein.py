from operator import itemgetter
from itertools import combinations
from random import random,randint,choice,sample
from time import time
from numpy import array
import cProfile
import pstats
from heapq import nlargest
from bottleneck import partsort,argpartsort,median



def f(peer,t):
	return peer[0]-t*peer[1]


def F(subset_values,t,k):
	l=[(v,f(v,t)) for v in subset_values]
	
	return map(itemgetter(0),sorted(l,key=lambda x:x[1],reverse=True)[:k])


def newton(values,k):
	T=values[:k]
	i=0
	prec_avg=weighted_average(T)
	while True:
		T_next=F(values,prec_avg,k)
		now_avg=weighted_average(T_next)
		if abs(prec_avg - now_avg)<10**(-9):
			break
		prec_avg=now_avg
		i+=1
	return T_next





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


# def random_epstein(values,k):
# 	s_values=set((x,y,i) for i,(x,y) in zip(range(len(values)),values))
# 	Aleft=A(s_values)
# 	#print Aleft
# 	#raw_input('...')
# 	Aright=float('inf')
# 	while len(s_values)>1:
# 		vrand=sample(s_values,1)[0]
# 		vrand_v=vrand[0]
# 		vrand_w=vrand[1]
# 		vrand_i=vrand[2]
# 		l=set([(vj,wj,vj-vrand_v,float('inf')) if wj==vrand_w else (vj,wj,wj-vrand_w,(vj-vrand_v)/(wj-vrand_w)) for (vj,wj,j) in s_values])
		
# 		E=set()
# 		X=set()
# 		Y=set()
# 		Z=set()
# 		for (vj,wj,delta,a) in l:
# 			if delta==0:
# 				E|={(vj,wj,delta,a)}
# 			elif (a<=Aleft and delta>0) or (a>=Aright and delta<0):
# 				X|={(vj,wj,delta,a)} 
# 			elif (a<=Aleft and delta<0) or (a>=Aright and delta>0):
# 				Y|={(vj,wj,delta,a)}
# 			else:
# 				Z|={(vj,wj,delta,a)}
		
# 		while True:
# 			amedian=sorted(Z,key=lambda x:x[3])[len(Z)/2][3]
# 			#print amedian
# 			F_A=F(l,amedian,k)	
# 			#F_Asum=sum(F_A)
# 			F_Asum=sum(f((v,w),amedian) for v,w,_,_ in F_A)
# 			if F_Asum==0:
# 				return amedian
# 			elif F_Asum>0:
# 				Aleft=amedian
# 			else:
# 				Aright=amedian


# 			for (vj,wj,delta,a) in l:
# 				if (a<=Aleft and delta>0) or (a>=Aright and delta<0):
# 					X|={(vj,wj,delta,a)} 
# 				elif (a<=Aleft and delta<0) or (a>=Aright and delta>0):
# 					Y|={(vj,wj,delta,a)}
# 				else:
# 					Z|={(vj,wj,delta,a)}


# 			if (len(X)+len(E)>=len(l)-k):
# 				nb_to_remove=max(len(E),len(X)+len(E)+k-len(l))
# 				l=l-sample(E,nb_to_remove)
# 				l=l-Y
# 				k=k-len(Y)-nb_to_remove
# 			elif (len(Y)+len(E)>=k):
# 				nb_to_collapse=max(len(E),len(Y)+len(E)-k)

# 			if len(Z)<=(len(values)/32.):
# 				break
# 	return l.pop()


# def random_epstein(values,k):
	
# 	S=set(values)
# 	Aleft=weighted_average(S)
# 	Aright=float('inf')
# 	while len(S)>1:
# 		#print len(S)
# 		sampled=sample(S,1)[0]
# 		vi=sampled[0]
# 		wi=sampled[1]
# 		E=set()
# 		X=set()
# 		Y=set()
# 		Z=set()
		
# 		for (vj,wj) in S:
# 			# if wj==wi:
# 			# 	delta_i_j=vj-vi
# 			# 	A_i_j=float('-inf')
# 			# else:
# 			# 	delta_i_j=wi-wj
# 			# 	A_i_j=(vi-vj)/(wi-wj)
# 			# if delta_i_j==0:
# 			# 	E|={(vj,wj)}
# 			# elif (A_i_j<=Aleft and delta_i_j>0) or (A_i_j>=Aright and delta_i_j<0):
# 			# 	X|={(vj,wj)} 
# 			# elif (A_i_j<=Aleft and delta_i_j<0) or (A_i_j>=Aright and delta_i_j>0):
# 			# 	Y|={(vj,wj)}
# 			if vj==vi and wj==wi:
# 				E|={(vj,wj)}
# 			elif f((vj,wj),Aleft)>=f((vi,wi),Aleft) and f((vj,wj),Aright)>=f((vi,wi),Aright):
# 				X|={(vj,wj)} 
# 			elif f((vj,wj),Aleft)<f((vi,wi),Aleft) and f((vj,wj),Aright)<f((vi,wi),Aright):
# 				Y|={(vj,wj)}
# 		Z=S-X-Y-E
# 		n=len(S)
# 		print '-----------'
# 		print X
# 		print Y
# 		print Z
# 		print E
# 		print '-----------'

# 		while True:
# 			A=median([(vi-vj)/(wi-wj) if wi!=wj else float('-inf') for (vj,wj) in Z])
# 			# if len(Z)==0:
# 			# 	break
# 			# A=sorted([(vi-vj)/(wi-wj) if wi!=wj else float('-inf') for (vj,wj) in Z])[len(Z)/2]#print A,Z
# 			#raw_input('MMMMMMMMMM')
# 			l=sorted([(vj-A*wj) for (vj,wj) in S],reverse=True)

# 			F_A=sum(l[:len(S)-k])
# 			print F_A
# 			if abs(F_A)<10**(-9):
# 				return A
# 			elif F_A>0:
# 				Aleft=A
# 			else:
# 				Aright=A

# 			#####################RECOMPUTE X,Y,Z#####################
# 			#E=set()
# 			X=set()
# 			Y=set()
# 			Z=set()
# 			E=set()
# 			# for (vj,wj) in S:
# 			# 	if wj==wi:
# 			# 		delta_i_j=vj-vi
# 			# 		A_i_j=float('-inf')
# 			# 	else:
# 			# 		delta_i_j=wi-wj
# 			# 		A_i_j=(vi-vj)/(wi-wj)

# 			# 	if (A_i_j<=Aleft and delta_i_j>0) or (A_i_j>=Aright and delta_i_j<0):
# 			# 		X|={(vj,wj)} 
# 			# 	elif (A_i_j<=Aleft and delta_i_j<0) or (A_i_j>=Aright and delta_i_j>0):
# 			# 		Y|={(vj,wj)}
# 			# Z=S-X-Y-E

# 			for (vj,wj) in S:
# 				if wj==wi:
# 					delta_i_j=vj-vi
# 					A_i_j=float('-inf')
# 				else:
# 					delta_i_j=wi-wj
# 					A_i_j=(vi-vj)/(wi-wj)
# 				if delta_i_j==0:
# 					E|={(vj,wj)}
# 				elif (A_i_j<=Aleft and delta_i_j>0) or (A_i_j>=Aright and delta_i_j<0):
# 					X|={(vj,wj)} 
# 				elif (A_i_j<=Aleft and delta_i_j<0) or (A_i_j>=Aright and delta_i_j>0):
# 					Y|={(vj,wj)}
# 			Z=S-X-Y-E

# 			#####################RECOMPUTE X,Y,Z#####################
# 			print 'X = ',len(X),'Y = ',len(Y),'Z = ',len(Z),'E = ',len(E),'S = ',len(S)
# 			raw_input('....')
# 			if ((len(X)+len(E))>=(len(S)-k)):
# 				nb_to_remove=min(len(E),max(len(E),len(X)+len(E)+k-len(S)))
# 				to_remove_E=set(sample(E,nb_to_remove))
				
# 				S=S-to_remove_E
# 				#E=E-to_remove_E
# 				S=S-Y
# 				k=k-len(Y)-nb_to_remove
		
# 				#Z=S-X-Y-E

# 			elif (len(Y)+len(E))>=k:
# 				nb_to_collapse=min(len(E),max(len(E),len(Y)+len(E)-k))
# 				values_to_collapse_E=set(sample(E,nb_to_collapse))
# 				#E=E-values_to_collapse_E
# 				values_to_collapse=values_to_collapse_E|X

# 				S=S-values_to_collapse
# 				#Z=Z-values_to_collapse
# 				#X=set()
				
# 				collapsed=(sum(x[0] for x in values_to_collapse),sum(x[1] for x in values_to_collapse))
# 				#X={collapsed}
# 				S=S|{collapsed}

# 				#Z=Z|{collapsed}

# 			if len(Z)<=n/4:
# 				print len(Z),len(S),len(E)
# 				break
# 			else:
# 				print len(Z),len(S),len(E)
# 				print Z


# 			raw_input('...')
# 	spop=S.pop()
# 	print 'hey ! '
# 	return spop[0]/spop[1]

def random_epstein(values,k):
	
	S=set(values)
	#print len(S),len(values)
	Aleft=weighted_average(S)
	Aright=float('inf')
	

	while len(S)>1:
		sampled=sample(S,1)[0]
		vi=sampled[0]
		wi=sampled[1]
		E=set();X=set();Y=set();Z=set()
	
		for (vj,wj) in S:
			if wj==wi:
				delta_i_j=vj-vi
				A_i_j=float('-inf')
			else:
				delta_i_j=wi-wj
				A_i_j=(vi-vj)/(wi-wj)


			if delta_i_j==0:
				E|={(vj,wj)}
			elif (A_i_j<=Aleft and delta_i_j>0) or (A_i_j>=Aright and delta_i_j<0):
				X|={(vj,wj)} 
			elif (A_i_j<=Aleft and delta_i_j<0) or (A_i_j>=Aright and delta_i_j>0):
				Y|={(vj,wj)}
		Z=S-X-Y-E
		n=len(S)
		# print '-----------'
		# print 'X = ',X
		# print 'Y = ',Y
		# print 'Z = ',Z
		# print 'E = ',E
		# print 'S = ',S
		# print '-----------'
		#raw_input('***************')
		while True:
			if len(Z)>0:
				#A=sorted([(vi-vj)/(wi-wj) if wi!=wj else float('-inf') for (vj,wj) in Z])[len(Z)/2]
				A=median([(vi-vj)/(wi-wj) if wi!=wj else float('-inf') for (vj,wj) in Z])
				l=sorted([f((vj,wj),A) for (vj,wj) in S],reverse=True)[:len(S)-k]
				
				#print l
				#raw_input('...')
				F_A=sum(l[:len(S)-k])
				
				if F_A==0:
					return A
				elif F_A>0:
					Aleft=A
				else:
					Aright=A
				#print [Aleft,Aright]
				
				#####################RECOMPUTE X,Y,Z#####################
				to_remove_from_z=set()
				for (vj,wj) in Z:
					delta_i_j=wi-wj
					A_i_j=(vi-vj)/(wi-wj)
					if (A_i_j<=Aleft and delta_i_j>0) or (A_i_j>=Aright and delta_i_j<0):
						X|={(vj,wj)} 
						to_remove_from_z|={(vj,wj)}
					elif (A_i_j<=Aleft and delta_i_j<0) or (A_i_j>=Aright and delta_i_j>0):
						Y|={(vj,wj)}
						to_remove_from_z|={(vj,wj)}
				Z=Z-to_remove_from_z
				#####################RECOMPUTE X,Y,Z#####################
				
				#print 'X = ',len(X),'Y = ',len(Y),'Z = ',len(Z),'E = ',len(E),'S = ',len(S)
				#raw_input('....')

			if ((len(X)+len(E))>=(len(S)-k)) and k>0:
				nb_to_remove=min(len(E),len(X)+len(E)-(len(S)-k))
				#print nb_to_remove,k,len(E)
				to_remove_E=set(sample(E,nb_to_remove))
				#print len(S)
				S=S-to_remove_E
				E=E-to_remove_E
				S=S-Y

				#print len(S)
				k=k-(len(Y)+nb_to_remove)
				Y=set()
				# print len(E),len(S)
				# raw_input('ooooo')
				# if k==0:
				# 	return weighted_average(S)

			elif (len(Y)+len(E))>=k:
				nb_to_collapse=min(len(E),len(Y)+len(E)-k)
				values_to_collapse_E=set(sample(E,nb_to_collapse))
				E=E-values_to_collapse_E
				values_to_collapse=values_to_collapse_E|X
				S=S-values_to_collapse
				collapsed=(sum(x[0] for x in values_to_collapse),sum(x[1] for x in values_to_collapse))
				X={collapsed}
				S=S|{collapsed}
			
			if len(Z)<=len(S)/32:
				break

	spop=S.pop()
	#print 'hey ! '
	return spop[0]/spop[1]




def COMPUTE_X_Y_Z_E(S,values,Aleft,Aright,vi,wi):
	E=set();X=set();Y=set();Z=set()
	E_add=E.add;X_add=X.add;Y_add=Y.add;Z_add=Z.add
	for j in S:
		vj,wj=values[j]
		if wj==wi:
			delta_i_j=vj-vi
			A_i_j=float('-inf')
		else:
			delta_i_j=wi-wj
			A_i_j=(vi-vj)/(wi-wj)


		if delta_i_j==0:
			E_add(j)
		elif (A_i_j<=Aleft and delta_i_j>0) or (A_i_j>=Aright and delta_i_j<0):
			X_add(j)
		elif (A_i_j<=Aleft and delta_i_j<0) or (A_i_j>=Aright and delta_i_j>0):
			Y_add(j)
		else:
			Z_add(j)
	return X,Y,Z,E



def UPDATE_X_Y_Z(S,values,Aleft,Aright,vi,wi,X,Y,Z):
	to_remove_from_z=set()
	X_add=X.add;Y_add=Y.add;Z_add=Z.add
	for j in Z:
		vj,wj=values[j]
		delta_i_j=wi-wj
		A_i_j=(vi-vj)/(wi-wj)
		if (A_i_j<=Aleft and delta_i_j>0) or (A_i_j>=Aright and delta_i_j<0):
			X_add(j) 
			to_remove_from_z.add(j)
		elif (A_i_j<=Aleft and delta_i_j<0) or (A_i_j>=Aright and delta_i_j>0):
			Y_add(j)
			to_remove_from_z.add(j)
	Z=Z-to_remove_from_z
	return X,Y,Z




def random_epstein_exact(values_input,nb_to_keep):
	nb_to_remove=len(values_input)-nb_to_keep
	k=nb_to_remove

	values=values_input[:]
	S=set(range(len(values)))
	n=len(S)
	indice_from_which_to_start=len(values)
	Aleft=weighted_average(values)
	
	Aright=float('inf')#max([x[0] for x in values])#float('inf')
	
	while len(S)>1:
		sampled=sample(S,1)[0]
		vi=values[sampled][0]
		wi=values[sampled][1]
		
		X,Y,Z,E=COMPUTE_X_Y_Z_E(S,values,Aleft,Aright,vi,wi)

		while True:
			#print 'HeLLo'
			if len(Z)>0:
				
				A=median([(vi-values[j][0])/(wi-values[j][1]) for j in Z])
				#A=[(vi-values[j][0])/(wi-values[j][1]) for j in Z][len(Z)/2]
				#print A
				
				#print [(vi-values[j][0])/(wi-values[j][1]) for j in Z],A
				l2=partsort(([A*values[j][1]-values[j][0] for j in S]),len(S)-k)[:len(S)-k]
				F_A=-sum(l2)
				
				if F_A==0: 

					return A

				elif F_A>0: 
					Aleft=A
				else: 
					Aright=A
				X,Y,Z=UPDATE_X_Y_Z(S,values,Aleft,Aright,vi,wi,X,Y,Z)



			if ((len(X)+len(E))>=(len(S)-k)) and k>0:
				nb_to_remove=min(len(E),len(X)+len(E)-(len(S)-k))
				to_remove_E=set(sample(E,nb_to_remove))
				S=S-to_remove_E
				E=E-to_remove_E
				S=S-Y
				#k=k-(len(Y)+nb_to_remove)
				k=k-(len(Y)+nb_to_remove)
				Y=set()

			elif (len(Y)+len(E))>=k:
				nb_to_collapse=min(len(E),len(Y)+len(E)-k)
				values_to_collapse_E=set(sample(E,nb_to_collapse))
				E=E-values_to_collapse_E
				values_to_collapse=values_to_collapse_E|X
				S=S-values_to_collapse

				collapsed_v=0.
				collapsed_w=0.

				for x in values_to_collapse:
					vx,wx=values[x]
					collapsed_v+=vx
					collapsed_w+=wx

				collapsed=(collapsed_v,collapsed_w)
				values.append(collapsed)
				X={indice_from_which_to_start}
				S.add(indice_from_which_to_start)
				indice_from_which_to_start+=1

			
			# if len(Z)<=len(S)/32:
			# 	break
			if len(Z)<=len(S)/32.:
				break

	spop=S.pop()
	#print values[spop]
	#raw_input('....')
	return values[spop][0]/values[spop][1]


def random_epstein_exact_weird_correction(values_input,nb_to_keep):
	nb_to_remove=len(values_input)-nb_to_keep
	k=nb_to_remove

	values=values_input[:]
	S=set(range(len(values)))
	n=len(S)
	indice_from_which_to_start=len(values)
	Aleft=weighted_average_weird_correction(values)
	
	Aright=float('inf')#max([x[0] for x in values])#float('inf')
	
	while len(S)>1:
		sampled=sample(S,1)[0]
		vi=values[sampled][0]
		wi=values[sampled][1]
		
		X,Y,Z,E=COMPUTE_X_Y_Z_E(S,values,Aleft,Aright,vi,wi)

		while True:
			#print 'HeLLo'
			if len(Z)>0:
				
				A=median([(vi-values[j][0])/(wi-values[j][1]) for j in Z])
				#A=[(vi-values[j][0])/(wi-values[j][1]) for j in Z][len(Z)/2]
				#print A
				
				#print [(vi-values[j][0])/(wi-values[j][1]) for j in Z],A
				l2=partsort(([A*values[j][1]-values[j][0] for j in S]),len(S)-k)[:len(S)-k]
				F_A=-sum(l2)
				
				if F_A==0: 

					return A

				elif F_A>0: 
					Aleft=A
				else: 
					Aright=A
				X,Y,Z=UPDATE_X_Y_Z(S,values,Aleft,Aright,vi,wi,X,Y,Z)



			if ((len(X)+len(E))>=(len(S)-k)) and k>0:
				nb_to_remove=min(len(E),len(X)+len(E)-(len(S)-k))
				to_remove_E=set(sample(E,nb_to_remove))
				S=S-to_remove_E
				E=E-to_remove_E
				S=S-Y
				#k=k-(len(Y)+nb_to_remove)
				k=k-(len(Y)+nb_to_remove)
				Y=set()

			elif (len(Y)+len(E))>=k:
				nb_to_collapse=min(len(E),len(Y)+len(E)-k)
				values_to_collapse_E=set(sample(E,nb_to_collapse))
				E=E-values_to_collapse_E
				values_to_collapse=values_to_collapse_E|X
				S=S-values_to_collapse

				collapsed_v=0.
				collapsed_w=1.

				for x in values_to_collapse:
					vx,wx=values[x]
					collapsed_v+=vx
					collapsed_w+=wx

				collapsed=(collapsed_v,collapsed_w)
				values.append(collapsed)
				X={indice_from_which_to_start}
				S.add(indice_from_which_to_start)
				indice_from_which_to_start+=1

			
			# if len(Z)<=len(S)/32:
			# 	break
			if len(Z)<=len(S)/32.:
				break

	spop=S.pop()
	#print values[spop]
	
	return values[spop][0]/values[spop][1]


def epstein_max_and_min_new(values,k,bottom=False):
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


if __name__ == '__main__':
	#V=[29.17, 35.75, 16.45, 2.26, 24.63, 46.92, 18.49, 19.85, 30.77, 28.69]
	#W=[1.93, 1.27, 1.53, 1.62, 1.02, 1.69, 1.89, 1.89, 1.83, 1.12]
	k=25
	n=100
	W=[float('%.2f'%(random()+1.01)) for _ in range(n)]
	V=[float('%.2f'%(random()*(200)+1)) for _ in range(n)]
	VW=[V[i]*W[i] for i in range(len(V))]
	values=[(x,y) for (x,y) in zip(VW,W)]
	#print values
	
	# s=time()
	# print newton_optimized(values,k),time()-s
	# s=time()
	# print random_epstein_exact(values,n-k),time()-s
	s=time()
	
	PROFILING=True
	if PROFILING:
		pr = cProfile.Profile()
		pr.enable()
	for k in range(n,n-80,-1):
		#print newton_optimized(values,k),time()-s
		print weighted_average(values)
		print n-k,epstein_max_and_min_new(values,k,False)#,time()-s#,newton_optimized(values,k)
		raw_input('...')
	if PROFILING:
		pr.disable()
		ps = pstats.Stats(pr)
		ps.sort_stats('cumulative').print_stats(20) #time

	
	# print W
	# print V
	#print weighted_average(newton(values,k))
	# s=time()
	# #objs=newton(values,5)
	# print weighted_average(newton(values,6)),weighted_average(newton(values,5)),weighted_average(newton(values,4)),weighted_average(newton(values,3)),weighted_average(newton(values,2))
	# print time()-s
	# print A(newton(values,5))
	# print max(A(s) for s in combinations(values,15))
