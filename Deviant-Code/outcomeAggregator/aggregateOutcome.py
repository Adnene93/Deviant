
from outcomeAggregator.aggregateOutcomeMeasures import aggregateOutcomeInitialize_DSC,aggregateOutcome_DSC,aggregateOutcomeFinalize_DSC

def compute_aggregates_outcomes(v_ids,u_ids,all_users_to_items_outcomes,outcome_tuple_structure,method_aggregation_outcome='STANDARD',outcomeTrack={}):
	
	# one_entitie=all_users_to_items_outcomes[next(all_users_to_items_outcomes.iterkeys())]
	# one_item=next(one_entitie.iterkeys())
	# outcome_tuple_structure=tuple(one_entitie[one_item])
	
	u_aggregated_to_items_outcomes={}
	u_agg_tuple=tuple(sorted(u_ids))
	if u_agg_tuple in outcomeTrack:
		return outcomeTrack[u_agg_tuple]
	else:
		for v in v_ids:
			vector_associated=aggregateOutcomeInitialize_DSC(outcome_tuple_structure, method_aggregation_outcome) 
			flag_at_least_someone_voted=False
			for u in u_ids:
				try:
					v_u=all_users_to_items_outcomes[u][v]
					flag_at_least_someone_voted=True
					vector_associated=aggregateOutcome_DSC(v_u, vector_associated, method_aggregation_outcome)
				except:
					continue
			if flag_at_least_someone_voted:
				u_aggregated_to_items_outcomes[v]=aggregateOutcomeFinalize_DSC(vector_associated,method_aggregation_outcome)
		outcomeTrack[u_agg_tuple]=u_aggregated_to_items_outcomes       

	return u_aggregated_to_items_outcomes