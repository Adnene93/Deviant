'''
Created on 13 juin 2017

@author: Adnene
'''


def aggregate_symbolic_vote(vectors_list,outcome_tuple_structure):  #just like parliament For against abstain
    mapping_res={}
    for v in vectors_list:
        val=v[0]
        if val not in mapping_res:
            mapping_res[val]=0
        mapping_res[val]+=1
    argmaxvote='';
    maxvotenb=-1
    for k in mapping_res:
        if maxvotenb<mapping_res[k]:
            maxvotenb=mapping_res[k]
            argmaxvote=k
    
    ret=(argmaxvote,)
    
    return ret


def aggregate_symbolic_vote_incremental_initialize(outcome_tuple_structure):
    return {}

def aggregate_symbolic_vote_incremental(user_vector,outcome_tuple): 
    #vector composed with only one value which is a rating value
    vote=user_vector[0]
    if vote not in  outcome_tuple:
        outcome_tuple[vote]=0
    outcome_tuple[vote]+=1
    return outcome_tuple

def aggregate_symbolic_vote_incremental_finalize(outcome_tuple): 
    argmaxvote='';
    maxvotenb=-1
    for k in sorted(outcome_tuple):
        if maxvotenb<outcome_tuple[k]:
            maxvotenb=outcome_tuple[k]
            argmaxvote=k
    
    ret=argmaxvote#(argmaxvote,)
    
    return ret

def aggregate_avgratings(vectors_list,outcome_tuple_structure): 
    #vector composed with only one value which is a rating value
    nb=0
    score=0.
    for v in vectors_list:
        score+=v[0]
        nb+=1
    ret=(score/float(nb),) if nb>0 else None
    return ret


def aggregate_avgratings_incremental_initialize(outcome_tuple_structure):
    return (0.,0.)

def aggregate_avgratings_incremental(user_vector,outcome_tuple): 
    #vector composed with only one value which is a rating value
    ret=((outcome_tuple[0]*outcome_tuple[1]+user_vector[0])/(outcome_tuple[1]+1),outcome_tuple[1]+1)
    return ret

def aggregate_avgratings_incremental_finalize(outcome_tuple): 
    return (outcome_tuple[0],)

def aggregate_avgratings_with_ponderation(vectors_list,outcome_tuple_structure): 
    #vector composed with two values which is a rating value and the number of voter in that group for that review
    nb=0
    score=0.
    for v in vectors_list:
        score+=v[0]*v[1]
        nb+=v[1]
    ret=(score/float(nb),float(nb)) if nb>0 else None
    return ret

def aggregate_avgratings_with_ponderation_incremental_initialize(outcome_tuple_structure):
    return (0.,0.)

def aggregate_avgratings_with_ponderation_incremental(user_vector,outcome_tuple): 
    #vector composed with two values which is a rating value and the number of voter in that group for that review
    ret=((outcome_tuple[0]*outcome_tuple[1]+user_vector[0]*user_vector[1])/(outcome_tuple[1]+user_vector[1]),outcome_tuple[1]+user_vector[1])
    return ret

def aggregate_avgratings_with_ponderation_incremental_finalize(outcome_tuple): 
    return outcome_tuple



def aggeragte_standard(vectors_list,outcome_tuple_structure): 
    len_outcome_tuple_structure=len(outcome_tuple_structure)
    range_len_outcome_tuple_structure=range(len_outcome_tuple_structure)
    ret=(0,)* len_outcome_tuple_structure
    for v in vectors_list:
        ret=tuple(ret[i]+v[i] for i in range_len_outcome_tuple_structure)
    return ret


def aggeragte_standard_incremental_initialize(outcome_tuple_structure):
    return tuple(outcome_tuple_structure)

def aggeragte_standard_incremental(user_vector,outcome_tuple): 
    len_outcome_tuple_structure=len(outcome_tuple)
    range_len_outcome_tuple_structure=range(len_outcome_tuple_structure)
    ret=tuple(outcome_tuple[i]+user_vector[i] for i in range_len_outcome_tuple_structure)
    
    return ret

def aggeragte_standard_incremental_finalize(outcome_tuple): 
    return outcome_tuple





def aggeragte_medoc_incremental_initialize(outcome_tuple_structure):
    return (0.,0.)

def aggeragte_medoc_incremental(user_vector,outcome_tuple): 
    #vector composed with two values which is a rating value and the number of voter in that group for that review
    ret=(outcome_tuple[0]+user_vector[0],outcome_tuple[1]+user_vector[1])
    return ret

def aggeragte_medoc_incremental_finalize(outcome_tuple): 
    return outcome_tuple


MAP_AGGREGATIONS_OUTCOME_FUNCTIONS={
    'SYMBOLIC_MAJORITY':aggregate_symbolic_vote_incremental,
    'AVGRATINGS_SIMPLE':aggregate_avgratings_incremental,
    'AVGRATINGS_PONDERATION':aggregate_avgratings_with_ponderation_incremental,
    'STANDARD':aggeragte_standard_incremental,
    'VECTOR_VALUES':aggeragte_standard_incremental,
    'MEDICAMENTS':aggeragte_medoc_incremental
}

MAP_AGGREGATIONS_OUTCOME_FINALIZE_FUNCTIONS={
    'SYMBOLIC_MAJORITY':aggregate_symbolic_vote_incremental_finalize,
    'AVGRATINGS_SIMPLE':aggregate_avgratings_incremental_finalize,
    'AVGRATINGS_PONDERATION':aggregate_avgratings_with_ponderation_incremental_finalize,
    'STANDARD':aggeragte_standard_incremental_finalize,
    'VECTOR_VALUES':aggeragte_standard_incremental_finalize,
    'MEDICAMENTS':aggeragte_medoc_incremental_finalize
}

MAP_AGGREGATIONS_OUTCOME_INITIALIZE_FUNCTIONS={
    'SYMBOLIC_MAJORITY':aggregate_symbolic_vote_incremental_initialize,
    'AVGRATINGS_SIMPLE':aggregate_avgratings_incremental_initialize,
    'AVGRATINGS_PONDERATION':aggregate_avgratings_with_ponderation_incremental_initialize,
    'STANDARD':aggeragte_standard_incremental_initialize,
    'VECTOR_VALUES':aggeragte_standard_incremental_initialize,
    'MEDICAMENTS':aggeragte_medoc_incremental_initialize
}


def aggregateOutcomeInitialize_DSC(outcome_tuple_structure,method='STANDARD'):
    
    return MAP_AGGREGATIONS_OUTCOME_INITIALIZE_FUNCTIONS[method](outcome_tuple_structure)

def aggregateOutcome_DSC(user_vector,outcome_tuple,method='STANDARD'):
    
    return MAP_AGGREGATIONS_OUTCOME_FUNCTIONS[method](user_vector,outcome_tuple)

def aggregateOutcomeFinalize_DSC(outcome_tuple,method='STANDARD'):
    
    return MAP_AGGREGATIONS_OUTCOME_FINALIZE_FUNCTIONS[method](outcome_tuple)