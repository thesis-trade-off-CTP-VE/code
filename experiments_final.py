# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:44:47 2021

@author: Timo van Donselaar
"""

import random
import copy

from pgmpy.models import BayesianModel
from pgmpy.readwrite import BIFReader
from pgmpy.readwrite import XMLBIF
from pgmpy.readwrite.XMLBeliefNetwork import XBNReader


from timeit import default_timer as timer
from datetime import timedelta

from CliqueTreePropagation import CliqueTreePropagation

class experiment_1:
    
    def __init__(self, model):
        self.model = model
        self.model_nodes = model.nodes
        self.model_nodes_shuffled = random.sample(self.model_nodes, len(self.model_nodes))        
        
        self.nodes_states = []
        for node in self.model_nodes:
            for cpd in model.get_cpds():
                can_break = False
                for key, value in cpd.state_names.items():
                    if node == key:
                        self.nodes_states.append((node, value))
                        can_break = True
                        break
                if can_break:
                    break
                        
        self.nodes_states_shuffled = random.sample(self.nodes_states, len(self.nodes_states))
        
    def test_without_evidence(self):
        analysis_ve_queries = []
        analysis_ctp_queries = []
        analysis_ctp_init = self.init_CTP()
        analysis_ctp_prop = self.CTP_global_prop()
        for query_node in random.sample(self.model_nodes, round(len(self.model_nodes)/2)):
            #VE(query_node)
            queried_fac_ve, analysis_ve = self.perform_VE(query_node)
            analysis_ve_queries.append(analysis_ve)
            print(queried_fac_ve)
            #CTP - marginalize(query_node)
            queried_fac_ctp, analysis_ctp_marg = self.CTP_marginalize(query_node)
            analysis_ctp_queries.append(analysis_ctp_marg)
            print(queried_fac_ctp)
        analysis_ve_queries_avg = self.average_analysis_queries(analysis_ve_queries)
        analysis_ctp_queries_avg = self.average_analysis_queries(analysis_ctp_queries)
        self.show_analysis(["CTP", "CTP", "CTP", "VE"], [analysis_ctp_init, analysis_ctp_prop, analysis_ctp_queries_avg, analysis_ve_queries_avg])
          
        
    def test_evidence_buildup(self, start_it, stop_it):
        model_nodes_left = list(self.model_nodes)
        evidence = []
        analysis_ctp_init = self.init_CTP()
        analysis_ctp_prop = self.CTP_global_prop()
        analysis_ctp_update_list = []
        analysis_ve_queries_avg_list = []
        analysis_ctp_queries_avg_list = []
        evidence_buffer = []
        for i in range(round(len(self.model_nodes)/2)):
        #for i in range(start_it, stop_it):
            #assign evidence
            node, states = self.nodes_states_shuffled[i]
            state_assign = random.sample(states, 1)[0]
            evidence.append((node, state_assign))
            model_nodes_left.remove(node)
            evidence_buffer.append((node, state_assign))
            if i % 4 == 0:
                #CTP - enter observation
                analysis_ctp_update = self.CTP_global_update(evidence_buffer)
                evidence_buffer.clear()
                analysis_ctp_update_list.append(analysis_ctp_update)
                
                analysis_ve_queries = []
                analysis_ctp_queries = []                    
                
                for query_node in random.sample(model_nodes_left, round(len(model_nodes_left)/2)):                
                    #VE(query_node)
                    queried_fac_ve, analysis_ve = self.perform_VE(query_node, evidence)
                    analysis_ve_queries.append(analysis_ve)
                    #CTP - marginalize(query_node)
                    queried_fac_ctp, analysis_ctp_marg = self.CTP_marginalize(query_node)
                    analysis_ctp_queries.append(analysis_ctp_marg)

                analysis_ve_queries_avg = self.average_analysis_queries(analysis_ve_queries)
                analysis_ctp_queries_avg = self.average_analysis_queries(analysis_ctp_queries)
                analysis_ve_queries_avg_list.append(analysis_ve_queries_avg)
                analysis_ctp_queries_avg_list.append(analysis_ctp_queries_avg)
        
        print("CTP init")        
        self.show_analysis(['CTP', 'CTP'], [analysis_ctp_init, analysis_ctp_prop])
        print()        
        self.show_analysis_list([analysis_ctp_update_list, analysis_ctp_queries_avg_list, analysis_ve_queries_avg_list])   
    
    
    def test_evidence_changes(self):
        analysis_ctp_init = self.init_CTP()
        analysis_ctp_prop = self.CTP_global_prop()
        analysis_ctp_retraction_list = []
        analysis_ve_queries_avg_list = []
        analysis_ctp_queries_avg_list = []

        for i in range(len(self.model_nodes)):
            print(i)
        
            model_nodes_left = list(self.model_nodes)
            evidence = []
            for node, states in random.sample(self.nodes_states, i):
                state_assign = random.sample(states, 1)[0]
                evidence.append((node, state_assign))
                model_nodes_left.remove(node)
            
            if i % 3 == 0:
                #CTP - global retraction; global update
                evidence_CTP = copy.deepcopy(evidence)
                analysis_ctp_retraction = self.CTP_global_retraction(evidence_CTP)
                analysis_ctp_retraction_list.append(analysis_ctp_retraction)
                
                analysis_ve_queries = []
                analysis_ctp_queries = []                    
                
                amount_of_queries = 0
                if len(model_nodes_left) > len(self.model_nodes)/2:
                    amount_of_queries = round(len(model_nodes_left)/2)
                else:
                    amount_of_queries = len(model_nodes_left)
                
                for query_node in random.sample(model_nodes_left, amount_of_queries):                
                    #VE(query_node)
                    queried_fac_ve, analysis_ve = self.perform_VE(query_node, evidence)
                    analysis_ve_queries.append(analysis_ve)
                    #CTP - marginalize(query_node)
                    queried_fac_ctp, analysis_ctp_marg = self.CTP_marginalize(query_node)
                    analysis_ctp_queries.append(analysis_ctp_marg)

                analysis_ve_queries_avg = self.average_analysis_queries(analysis_ve_queries)
                analysis_ctp_queries_avg = self.average_analysis_queries(analysis_ctp_queries)
                analysis_ve_queries_avg_list.append(analysis_ve_queries_avg)
                analysis_ctp_queries_avg_list.append(analysis_ctp_queries_avg)
        
        print("CTP init")        
        self.show_analysis(['CTP', 'CTP'], [analysis_ctp_init, analysis_ctp_prop])
        print()
        
        self.show_analysis_list([analysis_ctp_retraction_list, analysis_ctp_queries_avg_list, analysis_ve_queries_avg_list])
    
    
    def average_analysis_queries(self, analysis_queries):
        keys = analysis_queries[0].keys()
        result = {}
        for key in keys:
            sum_val = 0
            for analysis_query in analysis_queries:
                sum_val = sum_val + analysis_query[key]
            avg_val = sum_val/len(analysis_queries)
            result[key] = avg_val
        return result
    
    def show_analysis(self, model_types, analyses):
        for model_type, analysis in zip(model_types, analyses):
            for key, value in analysis.items():
                #print(key, "\t", value)
                #print(value, end=";")
                print(value)
        #print()
        
    def show_analysis_list(self, analyses_list):
        for i in range(len(analyses_list)):
            for j in range(len(analyses_list[i][0])):
                for l in range(len(analyses_list[i])):
                    key, value = list(analyses_list[i][l].items())[j]
                    print(value, end=" ; ")
                print()
            

    def perform_VE(self, query, evidence):
        from VariableElimination import VariableElimination
        queried_fac, analysis = VariableElimination.var_elim(VariableElimination, self.model, query, evidence)
        return queried_fac, analysis
        
    
    def init_CTP(self):        
        self.ctp = CliqueTreePropagation(self.model)
        start_build_tree = timer()
        self.ctp.build_clique_tree()
        end_build_tree = timer()
        
        start_init = timer()
        self.ctp.initialize_inference()
        end_init = timer()
        analysis = {"build tree" : end_build_tree-start_build_tree,
                    "amount of clusters" : len(self.ctp.clusters), 
                    "initialize inference" : end_init-start_init
                    }
        return analysis
                
    
    def CTP_global_prop(self, evidence = None):
        enter_obs_time = None
        if evidence != None:
            start_enter_obs = timer()
            self.ctp.enter_observation(evidence)
            end_enter_obs = timer()
            enter_obs_time = end_enter_obs-start_enter_obs
        start_init_prop = timer()
        self.ctp.global_prop()
        end_init_prop = timer()      
        analysis = {"enter observation" : enter_obs_time, 
                    "initial propagation" : end_init_prop-start_init_prop
                    }
        return analysis
        
    
    def CTP_global_update(self, evidence):
        start_update = timer()
        self.ctp.global_update(evidence)
        end_update = timer()
        analysis = {"global update" : end_update-start_update}
        return analysis
        
    
    def CTP_global_retraction(self, evidence=None):
        start_retraction_prop = timer()
        if evidence == None:
            self.ctp.global_retraction()
        else:
            self.ctp.global_retraction(evidence)
        self.ctp.global_prop()
        end_retraction_prop = timer()
        analysis = {"global retraction" : end_retraction_prop-start_retraction_prop}
        return analysis
        
    
    def CTP_marginalize(self, query):
        start_marginalize = timer()
        result = self.ctp.marginalize(query)
        end_marginalize = timer()
        analysis = {"marginalize" : end_marginalize - start_marginalize}
        return result, analysis
  
   
    
#reader = XMLBIF.XMLBIFReader("")
reader = BIFReader("")

model = reader.get_model()
experiment = experiment_1(model)
#experiment.test_without_evidence()
#experiment.test_evidence_buildup(0,1)
experiment.test_evidence_changes()

