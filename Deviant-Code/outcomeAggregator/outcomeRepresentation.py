'''
Created on 20 janv. 2017

@author: Adnene
'''
from math import sqrt
#from numba import jit

def outcome_representation_in_reviews(dataset,position_attribute,vector_of_outcome=None,method_aggregation_outcome='VECTOR_VALUES'): #[vector_of_outcome=['attr1,'attr2]]
    if vector_of_outcome is None or method_aggregation_outcome=='VECTOR_VALUES':
        vector_of_outcome=None
        vector_of_actions=tuple(sorted(set(x[position_attribute] for x in dataset)))  
        #print 'possible Outcomes are : ', vector_of_actions
        
    
    
    for x in dataset:
        if vector_of_outcome is None :
            x[position_attribute]={x[position_attribute]:1.}
        else :
            vector_of_actions=vector_of_outcome
            x[position_attribute]={f:x[f] for f in vector_of_actions}
        x[position_attribute]=tuple([x[position_attribute].get(pos,0.) for pos in vector_of_actions])
            
    return dataset 



        
        

        
    