'''
Created on 13 juin 2017

@author: Adnene
'''


def distance_nominal(o1,o2,domain_of_possible_outcomes=None,marginal_infos=None):
	return int(o1!=o2)

def distance_categorical(o1,o2,domain_of_possible_outcomes=None,marginal_infos=None):
	return int(o1!=o2)

def distance_ordinal_simple(o1,o2,domain_of_possible_outcomes=None,marginal_infos=None): #STILL NEED TO ADD INFO ABOUT THE MAXIMUM (My Ordinal)
	# print domain_of_possible_outcomes
	# print o1,o2,abs(o1-o2)/(domain_of_possible_outcomes[-1]-domain_of_possible_outcomes[0])
	# raw_input('....')
	return abs(o1-o2)/(domain_of_possible_outcomes[-1]-domain_of_possible_outcomes[0])

def distance_ordinal_complex(o1,o2,domain_of_possible_outcomes=None,marginal_infos=None): #STILL NEED TO ADD INFO ABOUT THE MAXIMUM (My Ordinal)
	# print domain_of_possible_outcomes
	# print o1,o2,abs(o1-o2)/(domain_of_possible_outcomes[-1]-domain_of_possible_outcomes[0])
	# raw_input('....')

	indice_of_o1=domain_of_possible_outcomes.index(o1)
	indice_of_o2=domain_of_possible_outcomes.index(o2)
		
	nc_plus_nk_divided=((marginal_infos[o1]+marginal_infos[o2])/2.)
	max_c_plus_max_k_divided=marginal_infos['nb_outcomes']-((domain_of_possible_outcomes[-1]+domain_of_possible_outcomes[0])/2.)
	


	return (sum((domain_of_possible_outcomes[g]-nc_plus_nk_divided)for g in range(indice_of_o1,indice_of_o2+1))/max_c_plus_max_k_divided)**2 