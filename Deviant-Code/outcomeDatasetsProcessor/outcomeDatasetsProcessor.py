'''
Created on 20 janv. 2017

@author: Adnene
'''
#from numba import jit
from util.csvProcessing import readCSVwithHeader
from outcomeAggregator.outcomeRepresentation import outcome_representation_in_reviews
from datetime import datetime
from filterer.filter import filter_pipeline_obj
from functools import partial
from operator import itemgetter
import gc
from bisect import bisect_left,bisect_right

def from_string_to_date(str_date,dateformat="%d/%m/%y"):
	return datetime.strptime(str_date, dateformat)


def process_outcome_dataset(itemsfile,usersfile,outcomesfile,numeric_attrs=[],array_attrs=[],outcome_attrs=None,method_aggregation_outcome='VECTOR_VALUES',
							itemsScope=[],users_1_Scope=[],users_2_Scope=[],ratings_to_remove=set(),nb_items=float('inf'),nb_individuals=float('inf'),attributes_to_consider=None,
							nb_items_entities=float('inf'),nb_items_individuals=float('inf'), hmt_to_itemset=False,
							delimiter='\t',replace_ids=True): 

	nb_itemsets_all_context=0
	nb_itemsets_all_individuals=0
	FULL_OUTCOME_CONSIDERED=False
	ITEMS_METADATA_NEEDED=False
	USE_CACHE=False
	SIZE_ESTIMATION=False
	SMALLER_DESCRIPTION_SPACE=True
	VERBOSE=False
	CLARIFY=True
	nb_outcome_considered=0


	AM_I_DEALING_WITH_PREAGGREGATED_DISTRIBUTIONS=False

	domain_of_possible_outcomes=set()
	if 'CACHE' in dir(process_outcome_dataset):
		items,items_header,users,users_header,outcomes,outcomes_header,items_id,users_id,outcome_attrs,position_attr,outcomes_processed,considered_items,users1,users2,(considered_items_ids),(considered_users_1_ids),(considered_users_2_ids),(considered_users_ids)=process_outcome_dataset.CACHE
	else:
		items,items_header=readCSVwithHeader(itemsfile,numberHeader=numeric_attrs,arrayHeader=array_attrs,selectedHeader=None,delimiter=delimiter)
		users,users_header=readCSVwithHeader(usersfile,numberHeader=numeric_attrs,arrayHeader=array_attrs,selectedHeader=None,delimiter=delimiter)
		outcomes,outcomes_header=readCSVwithHeader(outcomesfile,numberHeader=numeric_attrs,arrayHeader=array_attrs,selectedHeader=None,delimiter=delimiter)	
		items_id=items_header[0]
		users_id=users_header[0]
		
		outcome_attrs=outcome_attrs if outcome_attrs is not None else [outcomes_header[2]]
		position_attr=outcome_attrs[0]
		# outcomes_filtered=[];outcomes_filtered_append=outcomes_filtered.append
		# for row in outcomes:
		# 	if row[position_attr] not in ratings_to_remove:
		# 		outcomes_filtered_append(row)
		# outcomes=outcomes_filtered
		
		outcomes=[row for row in outcomes if row[position_attr] not in ratings_to_remove]

		if False:
			outcomes_processed=outcome_representation_in_reviews(outcomes,position_attr,outcome_attrs,method_aggregation_outcome)
		else:
			outcomes_processed=outcomes
		
		# ITEMS_COUNT_INDIVIDUALS_GIVEN_OUTCOMES={}
		# for row in outcomes_processed:
		# 	v_id_rev=row[items_id]
		# 	if v_id_rev not in ITEMS_COUNT_INDIVIDUALS_GIVEN_OUTCOMES:
		# 		ITEMS_COUNT_INDIVIDUALS_GIVEN_OUTCOMES[v_id_rev]=0
		# 	ITEMS_COUNT_INDIVIDUALS_GIVEN_OUTCOMES[v_id_rev]+=1
		
		# set_of_items_having_pairable_outcomes=set(map(itemgetter(0),(filter(lambda x:x[1]>1,ITEMS_COUNT_INDIVIDUALS_GIVEN_OUTCOMES.items())))) #Filtering entities having no pairable outcomes

		considered_items=filter_pipeline_obj(items, itemsScope)[0]
		#print len(considered_items),len({x[items_id] for x in considered_items})
		already_seen_ids=set()
		considered_items_final=[]
		for row in considered_items:
			if row[items_id] in already_seen_ids:
				continue
			considered_items_final.append(row)
			already_seen_ids|={row[items_id]}

		considered_items=considered_items_final
		# print len(considered_items)
		# raw_input('***')
		users1=filter_pipeline_obj(users, users_1_Scope)[0]
		users2=filter_pipeline_obj(users, users_2_Scope)[0]

		get_items_ids = partial(map,itemgetter(items_id))
		get_users_ids = partial(map,itemgetter(users_id))

		considered_items_ids=set(get_items_ids(considered_items))
		considered_users_1_ids=set(get_users_ids(users1))
		considered_users_2_ids=set(get_users_ids(users2))
		considered_users_ids=set(considered_users_1_ids)|set(considered_users_2_ids)
		if USE_CACHE:
			process_outcome_dataset.CACHE=[items,items_header,users,users_header,outcomes,outcomes_header,
										   items_id,users_id,outcome_attrs,position_attr,outcomes_processed,
										   considered_items[:],users1[:],users2[:],
										   set(considered_items_ids),set(considered_users_1_ids),set(considered_users_2_ids),set(considered_users_ids)]


	if nb_items<float('inf'):
		nb_items=int(nb_items)
		considered_items_ids=set(sorted(considered_items_ids)[:nb_items])
		considered_items=[x for x in considered_items if x[items_id] in considered_items_ids]
	if nb_individuals<float('inf'):
		nb_individuals=int(nb_individuals)
		considered_users_ids=set(sorted(considered_users_ids)[:nb_individuals])
		users1=[x for x in users1 if x[users_id] in considered_users_ids]
		users2=[x for x in users2 if x[users_id] in considered_users_ids]



	all_users_to_items_outcomes={}
	all_items_to_users_outcomes={}
	outcomes_considered=[];outcomes_considered_append=outcomes_considered.append
	items_metadata={row[items_id]:row for row in considered_items if row[items_id] in considered_items_ids} if ITEMS_METADATA_NEEDED else {}
	users_metadata={row[users_id]:row for row in users if row[users_id] in considered_users_ids}

	###############

	ITEMS_COUNT_INDIVIDUALS_GIVEN_OUTCOMES={}
	for row in outcomes_processed:
		if row[users_id] in considered_users_ids:
			v_id_rev=row[items_id]
			if v_id_rev in considered_items_ids:

				if v_id_rev not in ITEMS_COUNT_INDIVIDUALS_GIVEN_OUTCOMES:
					ITEMS_COUNT_INDIVIDUALS_GIVEN_OUTCOMES[v_id_rev]=0
				ITEMS_COUNT_INDIVIDUALS_GIVEN_OUTCOMES[v_id_rev]+=1
	set_of_items_having_pairable_outcomes=set(map(itemgetter(0),(filter(lambda x:x[1]>1,ITEMS_COUNT_INDIVIDUALS_GIVEN_OUTCOMES.items()))))
	considered_items_ids=considered_items_ids&set_of_items_having_pairable_outcomes
	###############


	for row in outcomes_processed:
		v_id_rev=row[items_id]
		u_id_rev=row[users_id]
		if v_id_rev in considered_items_ids and u_id_rev in considered_users_ids:
			pos_rev=row[position_attr]
			
			if u_id_rev not in all_users_to_items_outcomes:
				all_users_to_items_outcomes[u_id_rev]={}
			if v_id_rev not in all_items_to_users_outcomes:
				all_items_to_users_outcomes[v_id_rev]={}

			all_users_to_items_outcomes[u_id_rev][v_id_rev]=pos_rev
			all_items_to_users_outcomes[v_id_rev][u_id_rev]=pos_rev
			domain_of_possible_outcomes|={pos_rev}
			if FULL_OUTCOME_CONSIDERED: outcomes_considered_append({items_id:v_id_rev,users_id:u_id_rev,position_attr:pos_rev})
			nb_outcome_considered+=1

	considered_users_1_sorted=sorted(users1,key=itemgetter(users_id))
	considered_users_2_sorted=sorted(users2,key=itemgetter(users_id))
	considered_items_sorted=sorted(considered_items,key=itemgetter(items_id))
	considered_items_sorted=[row for row in considered_items_sorted if row[items_id] in considered_items_ids]

	if SIZE_ESTIMATION:
		from pympler.asizeof import asizeof
		print asizeof(all_users_to_items_outcomes)
		print asizeof(considered_items_sorted)
		print asizeof(considered_users_1_sorted)
		print asizeof(considered_users_2_sorted)
	gc.collect()


	if SMALLER_DESCRIPTION_SPACE:
		NB_SELECTED_ITEMSET_ENTITIES=nb_items_entities#100
		NB_SELECTED_ITEMSET_INDIVIDUALS=nb_items_individuals#100
		
		
		from enumerator.enumerator_attribute_complex import init_attributes_complex,create_index_complex
		from enumerator.enumerator_attribute_themes2 import tree_leafs,tree_remove
		

		#####################################ENTITIES - DESCRIPTION SPACE LIMITING##########################################
		concerned_attributes_entities_numeric=sorted(set(attributes_to_consider)&set(numeric_attrs)&set(items_header))
		concerned_attributes_entities_hmt=sorted(set(attributes_to_consider)&set(array_attrs)&set(items_header))
		concerned_attributes_entities_categorical=sorted(set(attributes_to_consider)&set(items_header)-(set(concerned_attributes_entities_hmt)|set(concerned_attributes_entities_numeric)))
		attributes_plain=[{'name':a,'type':'themes'} for a in concerned_attributes_entities_hmt]+[{'name':a,'type':'numeric'} for a in concerned_attributes_entities_numeric]+[{'name':a,'type':'simple'} for a in concerned_attributes_entities_categorical]
		attributes=[{'name':a,'type':'themes'} for a in concerned_attributes_entities_hmt]+[{'name':a,'type':'numeric'} for a in concerned_attributes_entities_numeric]+[{'name':a,'type':'simple'} for a in concerned_attributes_entities_categorical]
		
		attributes = init_attributes_complex(considered_items_sorted,attributes) #X
		index_all = create_index_complex(considered_items_sorted, attributes) #Y
		nb_itemsets_all=0
		for attr in attributes:
			if attr['type'] in {'numeric'}:
				nb_itemsets_all+=2*len(attr['domain'])
				attr['nb_items']=2*len(attr['domain'])
			else:
				nb_itemsets_all+=len(attr['domain'])
				attr['nb_items']=len(attr['domain'])
		
		if NB_SELECTED_ITEMSET_ENTITIES <= 1:
			NB_SELECTED_ITEMSET_ENTITIES = int(nb_itemsets_all * NB_SELECTED_ITEMSET_ENTITIES)
			print "NB_SELECTED_ITEMSET_ENTITIES AFTER RATIO = ", NB_SELECTED_ITEMSET_ENTITIES

		nb_itemsets_to_remove=max(0,nb_itemsets_all-NB_SELECTED_ITEMSET_ENTITIES)

		nb_itemsets_all_context=nb_itemsets_all
		

		if nb_itemsets_to_remove>0:
			nb_itemsets_all_context=NB_SELECTED_ITEMSET_ENTITIES
			if VERBOSE:
				print 'Entities Search Space Modification : ',NB_SELECTED_ITEMSET_ENTITIES
			factor=NB_SELECTED_ITEMSET_ENTITIES/float(nb_itemsets_all)
			#print factor
			for attr in attributes:
				attr['nb_items_new']=int(round(factor*attr['nb_items']))
				attr['nb_items_to_remove']=attr['nb_items']-attr['nb_items_new']
				attr_name=attr['name']
				attr_type=attr['type']
				attr_domain=attr['domain']
				attr_labelmap=attr['labelmap']
				if attr_type == 'themes':
					tags_removed=set(tree_remove(attr_domain,attr['nb_items_to_remove']))
					tags_keeped=set(attr_domain) - tags_removed
					for i,o in enumerate(index_all):
						o[attr_name]=o[attr_name] & tags_keeped
						considered_items_sorted[i][attr_name]=[attr_labelmap[t] if t!='' else t for t in sorted(o[attr_name])]
				elif attr_type == 'numeric':
					values_to_remove=int(attr['nb_items_to_remove']/2)
					values_to_keep=len(attr['domain'])-values_to_remove
					if values_to_keep==0:
						values_to_keep=1
					new_attr_domain = [attr_domain[int(round((k/float(values_to_keep)) * len(attr_domain)))] for k in range(values_to_keep)]
					mapto={x:new_attr_domain[bisect_right(new_attr_domain,x)-1] for x in attr_domain}
					
					for i,o in enumerate(index_all):
						o[attr_name]=mapto[o[attr_name]]
						considered_items_sorted[i][attr_name]=o[attr_name]
				elif attr_type == 'simple':
					if attr['nb_items_to_remove']>=len(attr['domain']):
						attr['nb_items_to_remove']=-1
					groupOfValues='From '+attr['domain'][0] + ' To ' + attr['domain'][attr['nb_items_to_remove']]
					new_attr_domain=[groupOfValues]+attr['domain'][attr['nb_items_to_remove']+1:]
					mapto={x:x if x in new_attr_domain else groupOfValues for x in attr_domain}
					for i,o in enumerate(index_all):
						o[attr_name]=mapto[o[attr_name]]
						considered_items_sorted[i][attr_name]=o[attr_name]
		else:
			if VERBOSE:
				print 'no Entities Search Space Modification'
		#####################################ENTITIES - DESCRIPTION SPACE LIMITING##########################################
		
		#####################################INDIVIDUALS - DESCRIPTION SPACE LIMITING##########################################
		concerned_attributes_individuals_numeric=sorted(set(attributes_to_consider)&set(numeric_attrs)&set(users_header))
		concerned_attributes_individuals_hmt=sorted(set(attributes_to_consider)&set(array_attrs)&set(users_header))
		concerned_attributes_individuals_categorical=sorted(set(attributes_to_consider)&set(users_header)-(set(concerned_attributes_individuals_hmt)|set(concerned_attributes_individuals_numeric)))
		attributes_plain=[{'name':a,'type':'themes'} for a in concerned_attributes_individuals_hmt]+[{'name':a,'type':'numeric'} for a in concerned_attributes_individuals_numeric]+[{'name':a,'type':'simple'} for a in concerned_attributes_individuals_categorical]
		attributes=[{'name':a,'type':'themes'} for a in concerned_attributes_individuals_hmt]+[{'name':a,'type':'numeric'} for a in concerned_attributes_individuals_numeric]+[{'name':a,'type':'simple'} for a in concerned_attributes_individuals_categorical]

		considered_users_sorted=users_metadata.values()

		attributes = init_attributes_complex(considered_users_sorted,attributes) #X
		index_all = create_index_complex(considered_users_sorted, attributes) #Y
		nb_itemsets_all=0
		for attr in attributes:
			if attr['type'] in {'numeric'}:
				nb_itemsets_all+=2*len(attr['domain'])
				attr['nb_items']=2*len(attr['domain'])
			else:
				nb_itemsets_all+=len(attr['domain'])
				attr['nb_items']=len(attr['domain'])
		if NB_SELECTED_ITEMSET_INDIVIDUALS <= 1:
			print "nb_itemsets_all",nb_itemsets_all,NB_SELECTED_ITEMSET_INDIVIDUALS
			NB_SELECTED_ITEMSET_INDIVIDUALS = int(nb_itemsets_all * NB_SELECTED_ITEMSET_INDIVIDUALS)
			print "NB_SELECTED_ITEMSET_INDIVIDUALS AFTER RATIO = ", NB_SELECTED_ITEMSET_INDIVIDUALS
		nb_itemsets_all_individuals=nb_itemsets_all
		nb_itemsets_to_remove=max(0,nb_itemsets_all-NB_SELECTED_ITEMSET_INDIVIDUALS)
		if nb_itemsets_to_remove>0:
			nb_itemsets_all_individuals=NB_SELECTED_ITEMSET_INDIVIDUALS
			if VERBOSE:
				print 'Individuals Search Space Modification : ',NB_SELECTED_ITEMSET_INDIVIDUALS
			factor=NB_SELECTED_ITEMSET_INDIVIDUALS/float(nb_itemsets_all)
			#print factor
			for attr in attributes:
				attr['nb_items_new']=int(round(factor*attr['nb_items']))
				attr['nb_items_to_remove']=attr['nb_items']-attr['nb_items_new']
				attr_name=attr['name']
				attr_type=attr['type']
				attr_domain=attr['domain']
				attr_labelmap=attr['labelmap']
				if attr_type == 'themes':
					tags_removed=set(tree_remove(attr_domain,attr['nb_items_to_remove']))
					tags_keeped=set(attr_domain) - tags_removed
					for i,o in enumerate(index_all):
						o[attr_name]=o[attr_name] & tags_keeped
						considered_users_sorted[i][attr_name]=[attr_labelmap[t] if t!='' else t for t in sorted(o[attr_name])]
				elif attr_type == 'numeric':
					values_to_remove=int(attr['nb_items_to_remove']/2)
					values_to_keep=len(attr['domain'])-values_to_remove
					if values_to_keep==0:
						values_to_keep=1
					new_attr_domain = [attr_domain[int(round((k/float(values_to_keep)) * len(attr_domain)))] for k in range(values_to_keep)]
					mapto={x:new_attr_domain[bisect_right(new_attr_domain,x)-1] for x in attr_domain}
					
					for i,o in enumerate(index_all):
						o[attr_name]=mapto[o[attr_name]]
						considered_users_sorted[i][attr_name]=o[attr_name]
				elif attr_type == 'simple':
					# print len(attr['domain'])
					# print attr['nb_items_to_remove']
					# print attr['nb_items_new']
					if attr['nb_items_to_remove']>=len(attr['domain']):
						attr['nb_items_to_remove']=-1
					groupOfValues='From '+attr['domain'][0] + ' To ' + attr['domain'][attr['nb_items_to_remove']]
					new_attr_domain=[groupOfValues]+attr['domain'][attr['nb_items_to_remove']+1:]
					mapto={x:x if x in new_attr_domain else groupOfValues for x in attr_domain}
					for i,o in enumerate(index_all):
						o[attr_name]=mapto[o[attr_name]]
						considered_users_sorted[i][attr_name]=o[attr_name]
		else:
			if VERBOSE:
				print 'no Individuals Search Space Modification'
		
		users1=[x for x in considered_users_sorted if x[users_id] in considered_users_1_ids]
		users2=[x for x in considered_users_sorted if x[users_id] in considered_users_2_ids]
		considered_users_1_sorted=sorted(users1,key=itemgetter(users_id))
		considered_users_2_sorted=sorted(users2,key=itemgetter(users_id))
		#####################################INDIVIDUALS - DESCRIPTION SPACE LIMITING##########################################

		items_metadata={row[items_id]:row for row in considered_items_sorted if row[items_id] in considered_items_ids} if ITEMS_METADATA_NEEDED else {}
		users_metadata={row[users_id]:row for row in considered_users_sorted if row[users_id] in considered_users_ids}
		
	# if CLARIFY:
	# 	concerned_attributes_entities_numeric=sorted(set(attributes_to_consider)&set(numeric_attrs)&set(items_header))
	# 	concerned_attributes_entities_hmt=sorted(set(attributes_to_consider)&set(array_attrs)&set(items_header))
	# 	concerned_attributes_entities_categorical=sorted(set(attributes_to_consider)&set(items_header)-(set(concerned_attributes_entities_hmt)|set(concerned_attributes_entities_numeric)))
	# 	attributes=[{'name':a,'type':'themes'} for a in concerned_attributes_entities_hmt]+[{'name':a,'type':'numeric'} for a in concerned_attributes_entities_numeric]+[{'name':a,'type':'simple'} for a in concerned_attributes_entities_categorical]
	# 	clarify_dataset(considered_items_sorted,attributes,items_id)
	#print considered_items_sorted[0]


	TOITEMSET=hmt_to_itemset
	if TOITEMSET:
		print '   '
		print 'Transform HMT to Itemset ...'
		print '   '
	if TOITEMSET:
		concerned_attributes_entities_numeric=sorted(set(attributes_to_consider)&set(numeric_attrs)&set(items_header))
		concerned_attributes_entities_hmt=sorted(set(attributes_to_consider)&set(array_attrs)&set(items_header))
		concerned_attributes_entities_categorical=sorted(set(attributes_to_consider)&set(items_header)-(set(concerned_attributes_entities_hmt)|set(concerned_attributes_entities_numeric)))
		attributes_plain=[{'name':a,'type':'themes'} for a in concerned_attributes_entities_hmt]+[{'name':a,'type':'numeric'} for a in concerned_attributes_entities_numeric]+[{'name':a,'type':'simple'} for a in concerned_attributes_entities_categorical]
		attributes=[{'name':a,'type':'themes'} for a in concerned_attributes_entities_hmt]+[{'name':a,'type':'numeric'} for a in concerned_attributes_entities_numeric]+[{'name':a,'type':'simple'} for a in concerned_attributes_entities_categorical]
		attributes = init_attributes_complex(considered_items_sorted,attributes) #X
		index_all = create_index_complex(considered_items_sorted, attributes) #Y
		for attr in attributes:
			#attr['nb_items_new']=int(round(factor*attr['nb_items']))
			#attr['nb_items_to_remove']=attr['nb_items']-attr['nb_items_new']
			attr_name=attr['name']
			attr_type=attr['type']
			attr_domain=attr['domain']
			attr_labelmap=attr['labelmap']
			index_tag_items={t:str(i).zfill(5) for i,t in enumerate(attr_domain.keys())}
			index_tag_items={k:v + " " +attr_labelmap[k].partition(" ")[-1] for k,v in index_tag_items.iteritems()}
			if attr_type == 'themes':
				for i,o in enumerate(index_all):
					# print '   '
					# print considered_items_sorted[i][attr_name]
					considered_items_sorted[i][attr_name]=[index_tag_items[t] for t in sorted(o[attr_name]) if t!='']
					# print considered_items_sorted[i][attr_name]
					# print '   '
					# raw_input('....')




	REPLACING_IDS=replace_ids
	if REPLACING_IDS:
		#print len(considered_items_ids),len(considered_items_sorted)
		#raw_input('xxx')
		ind_to_dict_items={i:x[items_id] for i,x in enumerate(considered_items_sorted)}
		dict_to_ind_items={v:k for k,v in ind_to_dict_items.iteritems()}
		considered_items_sorted=[dict([(items_id,dict_to_ind_items[x[items_id]])]+[(k,v) for k,v in x.items() if k !=  items_id]) for x in considered_items_sorted] 
		#print len(ind_to_dict_items),len(dict_to_ind_items)
		#raw_input('ccccccccccccc')
		


		ind_to_dict_users={i:x for i,x in enumerate(users_metadata)}
		dict_to_ind_users={v:k for k,v in ind_to_dict_users.iteritems()}
		considered_users_1_sorted=[dict([(users_id,dict_to_ind_users[x[users_id]])]+[(k,v) for k,v in x.items() if k !=  users_id]) for x in considered_users_1_sorted]
		considered_users_2_sorted=[dict([(users_id,dict_to_ind_users[x[users_id]])]+[(k,v) for k,v in x.items() if k !=  users_id]) for x in considered_users_2_sorted]
		considered_users_sorted=[dict([(users_id,dict_to_ind_users[x[users_id]])]+[(k,v) for k,v in x.items() if k !=  users_id]) for x in considered_users_sorted]
		all_users_to_items_outcomes={dict_to_ind_users[u]:{dict_to_ind_items[v]:o for v,o in u_votes.iteritems()} for u,u_votes in all_users_to_items_outcomes.iteritems()}
		all_items_to_users_outcomes={dict_to_ind_items[v]:{dict_to_ind_users[u]:o for u,o in v_votes.iteritems()} for v,v_votes in all_items_to_users_outcomes.iteritems()}
		items_metadata={row[items_id]:row for row in considered_items_sorted}
		users_metadata={dict_to_ind_users[u]:dict([(users_id,dict_to_ind_users[x[users_id]])]+[(k,v) for k,v in x.items() if k !=  users_id]) for u,x in users_metadata.iteritems()}
		
			

		considered_items_sorted=sorted(considered_items_sorted,key=lambda x:x[items_id])
		considered_users_1_sorted=sorted(considered_users_1_sorted,key=lambda x:x[users_id])
		considered_users_2_sorted=sorted(considered_users_2_sorted,key=lambda x:x[users_id])
		# print len(all_items_to_users_outcomes[4231]),ind_to_dict_items[4231],ITEMS_COUNT_INDIVIDUALS_GIVEN_OUTCOMES[ind_to_dict_items[4231]]
		# raw_input('....')

	

	if AM_I_DEALING_WITH_PREAGGREGATED_DISTRIBUTIONS:
		nb_outcome_considered=0.
		
		for u in all_users_to_items_outcomes:
			for e in all_users_to_items_outcomes[u]:
				all_users_to_items_outcomes[u][e]=eval(all_users_to_items_outcomes[u][e])
				all_items_to_users_outcomes[e][u]=eval(all_items_to_users_outcomes[e][u])
				nb_outcome_considered+=sum(all_users_to_items_outcomes[u][e])
		nb_outcome_considered=int(nb_outcome_considered)
		print "nb_outcome_considered : ",nb_outcome_considered
		domain_of_possible_outcomes=[iscore for iscore in range(len(all_users_to_items_outcomes[all_users_to_items_outcomes.keys()[0]][all_users_to_items_outcomes[all_users_to_items_outcomes.keys()[0]].keys()[0]]))]
		print "domain_of_possible_outcomes : ",domain_of_possible_outcomes
		for u in all_users_to_items_outcomes:
			for e in all_users_to_items_outcomes[u]:
				print u,e,all_users_to_items_outcomes[u][e]
				raw_input('....')

	process_outcome_dataset.STATS={
		'nb_items_entities':nb_itemsets_all_context,
		'nb_items_individuals':nb_itemsets_all_individuals,
		'items_header':items_header,
		'users_header':users_header,

	}
	return items_metadata,users_metadata,all_users_to_items_outcomes,all_items_to_users_outcomes,domain_of_possible_outcomes,outcomes_considered,items_id,users_id,considered_items_sorted,considered_users_1_sorted,considered_users_2_sorted,nb_outcome_considered





def clarify_dataset(considered_items_sorted,attributes_to_consider_entities,items_id):
	from enumerator.enumerator_attribute_complex import init_attributes_complex,create_index_complex
	from enumerator.enumerator_attribute_themes2 import tree_leafs,tree_remove
	attributes = init_attributes_complex(considered_items_sorted,attributes_to_consider_entities)
	index_all = create_index_complex(considered_items_sorted, attributes)
	

	mapping={}
	mapping_obj=[]
	considered_items_sorted_clarified=[]
	for i,o in enumerate(index_all):
		t_o=tuple(frozenset(o[a['name']]) if a['type']=='themes' else o[a['name']] for a in attributes)
		
		#raw_input('...')
		if t_o not in mapping:
			mapping[t_o]=set()
			mapping_obj.append(t_o)
			considered_items_sorted_clarified.append(considered_items_sorted[i])
			
			
		mapping[t_o]|={i}
		#print t_o,len(mapping)
	#print len(mapping), len(index_all)
	mapping_from_clarified_to_not_clarified={i:mapping[mapping_obj[i]] for i in range(len(mapping_obj))}
	for i in mapping_from_clarified_to_not_clarified:
		print i,mapping_from_clarified_to_not_clarified[i]
		raw_input('****')
	raw_input(('....'))

		 






# def process_outcome_dataset_with_time(itemsfile,usersfile,outcomesfile,
#                                      time_attr=None,time_format="%d/%m/%y",numeric_attrs=[],array_attrs=[],outcome_attrs=None,method_aggregation_outcome='VECTOR_VALUES',
#                                      delimiter='\t'):  #everything must be ordered by dateid
# 	items,items_header=readCSVwithHeader(itemsfile,numberHeader=numeric_attrs,arrayHeader=array_attrs,delimiter=delimiter)
# 	users,users_header=readCSVwithHeader(usersfile,numberHeader=numeric_attrs,arrayHeader=array_attrs,delimiter=delimiter)
# 	outcomes,outcomes_header=readCSVwithHeader(outcomesfile,numberHeader=numeric_attrs,arrayHeader=array_attrs,delimiter=delimiter)	
# 	items_id=items_header[0]
# 	users_id=users_header[0]
	
# 	outcome_attrs=outcome_attrs if outcome_attrs is not None else [outcomes_header[2]]
# 	position_attr=outcome_attrs[0]
# 	outcomes_processed=outcome_representation_in_reviews(outcomes,position_attr,outcome_attrs,method_aggregation_outcome)


	



# 	if time_attr is not None:
# 		for item in items:
# 			item[time_attr]=from_string_to_date(item[time_attr],time_format)
# 		items=sorted(items,key=lambda x:x[time_attr])
# 		map_items_to_sorted_id={v[items_id]:i for i,v in enumerate(items)} 
# 		items_metadata={}
# 		for item in items:
# 			item[items_id]=map_items_to_sorted_id[item[items_id]]
# 			items_metadata[item[items_id]]=item
# 		for row in outcomes_processed:
# 			row[items_id]=map_items_to_sorted_id[row[items_id]]
# 	else:
# 		items_metadata={row[items_id]:row for row in items}

	
		
# 	users_metadata={row[users_id]:row for row in users}


# 	all_users_to_items_outcomes={}
# 	for row in outcomes_processed:
# 		v_id_rev=row[items_id]
# 		u_id_rev=row[users_id]
# 		pos_rev=row[position_attr]
# 		if u_id_rev not in all_users_to_items_outcomes:
# 			all_users_to_items_outcomes[u_id_rev]={}
# 		all_users_to_items_outcomes[u_id_rev][v_id_rev]=pos_rev

# 	return items_metadata,users_metadata,all_users_to_items_outcomes,items_id,users_id