import numpy as np
import csv
from bisect import bisect_left


def histograms(qualities,starting_point=-5,width=0.01,lim_max=5.):
	nb_histograms=int((lim_max-starting_point)/width)
	x_axis_ticks=[];possible_intervals=[];all_data_points=[]
	nb_hists=[]
	for i in range(starting_point,nb_histograms):
		x_axis_ticks.append(starting_point+i*width)
		all_data_points.append([])
		nb_hists.append(0)
		possible_intervals.append([i*width,(i+1)*width])
	for q in qualities:
		nb_hists[bisect_left(x_axis_ticks,q)-1]+=1
		all_data_points[bisect_left(x_axis_ticks,q)-1].append(q)

	sum_nb_hists=float(sum(nb_hists))
	nb_hists_freq=[x/sum_nb_hists for x in nb_hists]
	print sum_nb_hists

	for i in range(len(nb_hists)):
		print possible_intervals[i], ' : ', nb_hists[i]#, '  - ', all_data_points[i]
	#raw_input('next')
	return x_axis_ticks,nb_hists_freq


def write_a_normal_distrbution():
	destination="./distribution.csv"
	data=[x for x in list(np.random.normal(0,1, 1000))]
	x,y=histograms(data)
	print y
	raw_input('...')

	with open(destination, 'wb') as f:
		writer = csv.writer(f,delimiter="\t")
		
		
		writer.writerows([[a] for a in y])

write_a_normal_distrbution()