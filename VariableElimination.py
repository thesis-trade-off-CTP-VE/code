# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:14:56 2021

@author: Timo van Donselaar
"""

from timeit import default_timer as timer
from datetime import timedelta

class VariableElimination:
    
    from pgmpy.models import BayesianModel
    from pgmpy.inference.EliminationOrder import WeightedMinFill
    import copy    
       
    
    def get_variable_cpd(self, cpd):
        return cpd.variable
    
    def get_node_for_variable(l, value):
        for node in l:
            if node[0][0] == value:
                return node
        # Matches behavior of list.index
        raise ValueError("list.index(x): x not in list")
        
    def get_parents(self, cpd):
        variables = self.copy.deepcopy(cpd.variables)
        variables.remove(cpd.variable)
        return variables
    
    def get_children(cpds, parent_var):
        children = []
        for cpd in cpds:
            if not cpd.variable == parent_var:
                if parent_var in cpd.variables:
                    children.append(cpd.variable)
        return children    
    
    """
    The Bayes ball algorithm for finding requisite probability nodes (relevant 
    nodes), requisite observation nodes and irrelevant nodes.
    """
    def bayes_ball(self, nodes, queried_var, evidence=None):
    
        # variable, cpd, visited, marked on the top, marked on the bottom
        nodes_markers = [list(a) for a in zip(self.copy.deepcopy(nodes), 
                              [False]*len(nodes), [False]*len(nodes), [False]*len(nodes))]
        # false = visit from chid ; true = visit from parent
        schedule = [(self.get_node_for_variable(nodes_markers, queried_var), False)]
        
        relevant_observed = []
        relevant_nodes = []
        irrelevant = []
        
        while len(schedule)>0:
            j = schedule.pop(0)
            j_in_evidence = False
            j[0][1] = True        
            if j[0][0][4] != None:
                j_in_evidence = True
                relevant_observed.append(j[0][0][0])        
            if not j_in_evidence and j[1] == False:
                if not j[0][2]:
                    j[0][2] = True
                    relevant_nodes.append(j[0][0][0])
                    for parent in j[0][0][2]:
                        parent_node  = self.get_node_for_variable(nodes_markers, parent)
                        schedule.append((parent_node, False))
                if not j[0][3]:
                    j[0][3] = True
                    for child in j[0][0][3]:
                        child_node = self.get_node_for_variable(nodes_markers, child)
                        schedule.append((child_node, True))
            if j[1] == True:
                if j_in_evidence and not j[0][2]:
                    j[0][2] = True
                    relevant_nodes.append(j[0][0][0])
                    for parent in j[0][0][2]:
                        parent_node  = self.get_node_for_variable(nodes_markers, parent)
                        schedule.append((parent_node, False))
                if not j_in_evidence and not j[0][3]:
                    j[0][3] = True
                    for child in j[0][0][3]:
                        child_node = self.get_node_for_variable(nodes_markers, child)
                        schedule.append((child_node, True))            
        
        for node in nodes_markers:
            if not node[3]:
                irrelevant.append(node[0][0])
        
        return (irrelevant, relevant_nodes, relevant_observed)  
    
       
    """
    Function for finding barren nodes (not used)
    """
    def find_barren(nodes, queried_var, evidence = None):
        check_barren = []
        barren = []
        for node in nodes:
            if node[0] != queried_var:
                if len(node[3]) == 0:
                    if node[4] == None:
                        barren.append(node[0])
                        for parent in node[2]:
                            check_barren.append(parent)
    
        while len(check_barren)>0:
            check_node = check_barren.pop(0)
            if check_node[0] != queried_var:
                if all(x in barren for x in check_node[3]) and node[4] == None:
                    barren.append(check_node[0])
                    for parent in check_node[2]:
                        check_barren.append(parent)
        
        return barren
    
    
    """
    The variable elimination algorithm (included pruning with the Bayes ball
    algorithm).
    """
    def var_elim(self, model, queried_var, evidence=None):
        
        # get conditional probabilities from the model
        cpds = model.get_cpds()
#        # copy the list of variables, to the list of variables to be eliminated             
#        variables = self.copy.deepcopy(vars_model)      
        
        #variables = model.nodes
        variables = [self.get_variable_cpd(self, cpd) for cpd in cpds]
        
        #variable, cpd, parents, children, evidence
        nodes = [list(a) for a in zip(self.copy.deepcopy(variables), 
                      self.copy.deepcopy(cpds), 
                      [self.get_parents(self, x) for x in self.copy.deepcopy(cpds)],
                      [self.get_children(cpds, x) for x in self.copy.deepcopy(variables)], 
                      [None]*len(cpds))]
        
        start_total = timer()
            
        if evidence != None:
            for node in nodes:        
                for single_evidence in evidence:
                    if single_evidence[0] == node[0]:
                        node[4] = single_evidence                                 
        
        start_prune = timer()
        
        irrelevant, relevant_nodes, relevant_obs = self.bayes_ball(self, nodes, queried_var, evidence)
        to_prune = [x for x in variables if not x in relevant_nodes]
        
        end_prune = timer()
        
        #barren = self.find_barren([node for node in nodes if not node[0] in to_prune], queried_var, evidence)
        #to_prune.extend(barren)
        
        #print(irrelevant)
        #print(relevant_nodes)
        #print(relevant_obs)
        #print(barren)
        #print(to_prune)
        
        variables = [x for x in variables if x not in to_prune] 
        
        #time for this subprocess is ignored
        start_evidence = timer()
        
        # make a list with (possibly reduced) factors
        factors = []       
        for node in nodes:
            if node[0] not in to_prune:
                factor = node[1].to_factor()
                if evidence != None:
                    for single_evidence in evidence:
                        if single_evidence[0] in factor.scope():
                            factor.reduce([single_evidence], inplace=True)
                        if single_evidence[0] in variables:
                            # an observed variabe does not need to be eliminated
                            variables.remove(single_evidence[0])
                factors.append(factor)   
        
        #time for this subprocess is ignored        
        end_evidence = timer()        
            
        # the queried variable should not be eliminated
        variables.remove(queried_var)
        # get an elimination ordering
        elim_order = self.WeightedMinFill(model).get_elimination_order(variables)
    #    print(elim_order)
        
        start_elim = timer()
    
        nr_multiplications = 0
        nr_sum_out = 0
        
        # perform the elimination
        for i in range(len(elim_order)):
            # initialize factor that is the factor after multiplication
            factor_new = None
            # initialize list with factors that are multiplied an can be removed
            factors_to_remove = []
            for factor in factors:
                # if the factor contains the current variable to be removed
                if elim_order[i] in factor.scope():
                    # if the mulitplication-factor (product) is not yet specified
                    if factor_new==None:
                        # then fill the multiplication-factor with ones
                        factor_new = factor.identity_factor()
                    # multiply the multiplication-factor with the factor
                    factor_new.product(factor, inplace=True)
                    nr_multiplications = nr_multiplications + 1
                    # add current factor to the to-be-removed list
                    factors_to_remove.append(factor)
            # chech wheter multiplication-factor is specified (in that case there
            # where some relevant factors)
            if factor_new != None:
                # remove the to-be-removed factors from the list of factors
                factors[:] = [f for f in factors if f not in factors_to_remove]
                # sum out (marginalize) the to-be-eliminated variable from the 
                # multiplication-factor
                factor_new.marginalize([elim_order[i]], inplace=True)
                nr_sum_out = nr_sum_out + 1
                # add resultant factor to the list of factors
                factors.append(factor_new)
            
        # all the factors left should now contain only the queried variable
        # to get the proper probabilities of this variable all factors left are
        # multiplied and the resulting factor is normalized
        factor_result = None
        for factor in factors:
            if factor_result==None:
                factor_result = factor.identity_factor()
            factor_result.product(factor, inplace=True)
        factor_result.normalize()
        
        end_elim = timer()
        end_total = timer()     
        
#        print("total time:", timedelta(seconds=end_total-start_total))
#        print("time evidence:", timedelta(seconds=end_evidence-start_evidence))
#        print("time prune:", timedelta(seconds=end_prune-start_prune))
#        print("time elimination:", timedelta(seconds=end_elim-start_elim))
#        print("number of multiplications:", nr_multiplications)
#        print("number of marginalizations:", nr_sum_out)
        analysis = {"total" : end_total-start_total, 
                    "evidence" : end_evidence-start_evidence, 
                    "prune": end_prune-start_prune,
                    "elimination" : end_elim-start_elim,
                    "multiplications" : nr_multiplications,
                    "marginalizations" : nr_sum_out}
            
        return factor_result, analysis
    
    
    
    
