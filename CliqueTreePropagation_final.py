# -*- coding: utf-8 -*-
"""
@author: Timo van Donselaar
"""

class CliqueTreePropagation:
    
    from pgmpy.models import BayesianModel
    from pgmpy.inference.EliminationOrder import WeightedMinFill
    from pgmpy.factors.discrete import DiscreteFactor
    import copy
    import operator
    
    """
    Initialize this class
    Define several lists for the variables, cpds, etc.
    """
    def __init__(self, model):
        self.model = model
        self.cpds = self.copy.deepcopy(model.get_cpds())  
        self.variables = [self.get_variable_cpd(cpd) for cpd in self.cpds]
        self.nodes_parents = [list(a) for a in zip(self.copy.deepcopy(self.variables), 
                            self.copy.deepcopy(self.cpds),
                            [model.get_parents(x) for x in self.variables])]
        self.nodes3 = [list(a) for a in zip(self.copy.deepcopy(self.variables), 
                            self.copy.deepcopy(self.cpds))]
        self.evidence = []
        self.cardinality_nodes = model.get_cardinality()
        self.nodes_states = []
        for node in self.variables:
            for cpd in self.cpds:
                can_break = False
                for key, value in cpd.state_names.items():
                    if node == key:
                        self.nodes_states.append((node, value))
                        can_break = True
                        break
                if can_break:
                    break
        self.nodes_states = dict(self.nodes_states)
    
    """
    Build a clique tree by first calling the moralize function, that gives some
    filler edges, and then calling the triangulate function, that gives the 
    clusters (cliques). Then the sepsets can be defined.
    """
    def build_clique_tree(self):
        self.fill_edges = self.moralize(self.nodes_parents, self.model)
        self.edges = list(self.model.edges)
        self.edges.extend(self.fill_edges)
        self.clusters = self.triangulate(self.variables, self.model, self.edges)
        self.sepsets = self.find_sepsets(self.clusters)
        
    """
    Initialize inference by making a list with cluster-factor pairs. 
    The factors of the variables (conditional probability table) are 
    assimilated in the factors of the clusters. Another list it made with
    cluster-factor-marker triplets.
    """
    def initialize_inference(self):
        self.clusters_factors = []
        self.clusters_factors.clear()
        nodes_list = self.copy.deepcopy(self.nodes3)
        for cluster in self.clusters:
            cardinalities = []
            total_cardinality = 1
            state_names = {}
            for node in cluster:
                cardinalities.append(self.cardinality_nodes[node])
                total_cardinality = total_cardinality * self.cardinality_nodes[node]
                state_names[node] = self.nodes_states[node]
            factor = self.DiscreteFactor(list(cluster), cardinalities, [1]*total_cardinality, state_names)
            self.clusters_factors.append([cluster, factor])
        for node in nodes_list:
            factor = self.copy.deepcopy(node[1].to_factor())
            for cluster_factor in self.clusters_factors:
                if set(factor.variables).issubset(cluster_factor[0]):
                    cluster_factor[1].product(factor, inplace=True)
                    break #break inner loop 
        # cluster_factor: [set/cluster, factor]
        self.clusters_fac_mark = []
        self.clusters_fac_mark.clear()
        for cluster_factor in self.clusters_factors:
            self.clusters_fac_mark.append([cluster_factor[0], cluster_factor[1], False])
        # cluster_fac_mark: [set/cluster, factor, marker]
    
    """
    Global propagation is performed by first assigning a root cluster 
    (cluster_x), unmarking all clusters, calling collect evidence with
    cluster_x, unmarking all clusters and then calling distribute evidence with
    cluster_x.
    """
    def global_prop(self):
        cluster_x = self.clusters_fac_mark[round(len(self.clusters_fac_mark)/2)]
        for cluster in self.clusters_fac_mark:
            cluster[2] = False
        self.collect_evidence(cluster_x)
        for cluster in self.clusters_fac_mark:
            cluster[2] = False
        self.distribute_evidence(cluster_x)        
    
    """
    For this function a cluster containing queried_var is looked for and from
    this cluster all other variable are summed out (marginalized). The result
    is then normalized.
    """
    def marginalize(self, queried_var):
        for cluster_fac_mark in self.clusters_fac_mark:
            if queried_var in cluster_fac_mark[0]:
                result_prop = cluster_fac_mark[1].marginalize(list(cluster_fac_mark[0]-{queried_var}), inplace=False)
        result_prop.normalize(inplace=True)
        return result_prop
    
    """
    Enter observations for a variables that are not yet observed. 
    """
    def enter_observation(self, evidence):
        for var_1, _ in evidence:
            for  var_2, _ in self.evidence:
                if var_1 == var_2:
                    raise ValueError("already observation for (some of) the variables")
        self.evidence.extend(evidence)
        evidence_to_process = evidence
        cluster_no = 0
        clusters_changed = 0
        while len(evidence_to_process)>0 and cluster_no < len(self.clusters_fac_mark):
            cluster = self.clusters_fac_mark[cluster_no]
            cluster_changed = False
            cluster_no = cluster_no + 1            
            factor = cluster[1]        
            evidence_processed = []
            for var, state_name in evidence_to_process:
                if var in cluster[0]:
                    values_to_zero = [(var, factor.get_state_no(var, name)) for name in factor.state_names[var] if name != state_name]
                    slice_ = [slice(None)] * len(factor.variables)
                    for var, state in values_to_zero:
                        var_index = factor.variables.index(var)
                        slice_[var_index] = state
                        axis = factor.values[tuple(slice_)].ndim
                        if axis==0:
                            factor.values[tuple(slice_)] = 0
                        else:
                            self.set_to_zero(factor.values[tuple(slice_)], axis)
                    evidence_processed.append((var, state_name))
                    cluster_changed = True
            evidence_to_process[:] = [e for e in evidence_to_process if e not in evidence_processed]
            if cluster_changed:
                clusters_changed = clusters_changed + 1
        return (clusters_changed, cluster_no)
    
    """
    Perform a global_update by calling enter_observation with new_evidence, 
    which will return the clusters that are changed and the number of changed
    clusters. When only one cluster is changed, only a distribution of 
    evidence follows, otherwise a global propagation follows.
    """
    def global_update(self, new_evidence):
        clusters_changed, cluster_no = self.enter_observation(new_evidence)
        if clusters_changed == 1:
            for cluster in self.clusters_fac_mark:
                cluster[2] = False            
            self.distribute_evidence(self.clusters_fac_mark[cluster_no-1])
        else:
            self.global_prop()
            
    """
    Perform a global retraction by clearing the current evidence, calling
    initialize_inference() and deleting the factors of the sepsts. Possibly
    some new observations are entered.
    """
    def global_retraction(self, new_evidence=None):
        self.evidence.clear()
        self.initialize_inference()
        for sepset in self.sepsets:
            if len(sepset) > 3:
                del sepset[3]
        if new_evidence != None:
            self.enter_observation(new_evidence)
        
    """
    Get the variable of the cpd.
    """
    def get_variable_cpd(self, cpd):
        return cpd.variable
    
    """
    Get the node with a certain variable.
    """
    def get_node_for_variable(self, l, value):
        for node in l:
            if node[0] == value:
                return node
        # Matches behavior of list.index
        raise ValueError("list.index(x): x not in list")
        
    """
    Get the parents of a variable with a certain cpd.
    """
    def get_parents(self, cpd):
        variables = self.copy.deepcopy(cpd.variables)
        variables.remove(cpd.variable)
        return variables
    
    """
    Get the children of some parent variable.
    """
    def get_children(self, cpds, parent_var):
        children = []
        for cpd in cpds:
            if not cpd.variable == parent_var:
                if parent_var in cpd.variables:
                    children.append(cpd.variable)
        return children  
    
    """
    Define some filler edges to moralize the network.
    """
    def moralize(self, nodes, model):
        fill_edges = []
        for node in nodes:
            if len(node[2]) > 1:
                for i in range(len(node[2])-1):
                    for j in range(i+1,len(node[2])):
                        fill_edges.append((node[2][i], node[2][j]))
        return fill_edges 
    
   
    def get_neighbors(self, edges, variable):
        neighbors = []    
        for edge in edges:
            if variable in edge:
                neighbors.append(edge[abs(edge.index(variable) - 1)])
        return neighbors 
        
    
    def triangulate(self, variables, model, edges):
        clusters = []
        elim_order = self.WeightedMinFill(self.model).get_elimination_order(variables)
        eliminated = []
        for variable in elim_order:
            cluster = []
            neighbors = self.get_neighbors(edges, variable)
            cluster.append(variable)
            for neighbor in neighbors:
                if neighbor not in eliminated:
                    cluster.append(neighbor)                              
            
            fill_edges = []
            for i in range(len(cluster)-1):
                for j in range(i+1,len(cluster)):
                    if not cluster[j] in self.get_neighbors(edges, cluster[i]):
                        fill_edges.append((cluster[i], cluster[j]))
            edges.extend(fill_edges)
                    
            cluster = set(cluster)
            if not any(cluster.issubset(c) for c in clusters):
                clusters.append(cluster)
            eliminated.append(variable)
        return clusters
    
    
    def mass_sepset(self, sepset):
        return len(sepset[0])
    
    
    def weight_sepset(self, sepset):
        weight = 1
        for variable in sepset[0]:
            node = self.get_node_for_variable(self.nodes3, variable)
            single_weight = len(node[1].get_values())
            weight = weight * single_weight
        return weight
    
    
    def find_sepsets(self, clusters):
        sepsets_all = []
        sepsets_final = []
        forest  = [[cluster] for cluster in clusters]
            
        for i in range(len(clusters)-1):
            for j in range(i+1, len(clusters)):
                if len(clusters[i].intersection(clusters[j])) > 0:
                    sepset = [clusters[i].intersection(clusters[j]), 
                          clusters[i], clusters[j]]
                    sepsets_all.append(sepset)
        sepsets_plus = [ [s[0], s[1], s[2], (-self.mass_sepset(s), self.weight_sepset(s))] for s in sepsets_all]
        sepsets_plus.sort(key = self.operator.itemgetter(3))
        for sepset in sepsets_plus:
            tree1 = None
            tree2 = None
            for tree in forest:
                if sepset[1] in tree:
                    tree1 = tree
            for tree in forest:
                if sepset[2] in tree:
                    tree2 = tree
            if tree1 != tree2:
                sepsets_final.append([sepset[0], sepset[1], sepset[2]])
                tree1.extend(tree2)
                forest.remove(tree2)
            
        return sepsets_final
    
    
    def get_cluster_fac_mark(self, cluster_set):
        for cluster_fac_mark in self.clusters_fac_mark:
            if cluster_fac_mark[0] == cluster_set:
                return cluster_fac_mark    
    
    
    def pass_message(self, cluster_x, sepset_r, cluster_y):        
        if len(sepset_r) > 3:
            r_old = sepset_r[3]
        r_new = cluster_x[1].marginalize(list(cluster_x[0] - sepset_r[0]), inplace=False)
        
        if len(sepset_r) > 3:
            r_change = r_new.divide(r_old, inplace=False)
            sepset_r[3] = r_new
        else:
            r_change = r_new
            sepset_r.append(r_new)

        cluster_y[1].product(r_change, inplace=True)
        
        # sepset is appended to: 
        # [(sep)set being intersection of X and Y, set/cluster X, set/cluster Y, 
        #   factor of intersection of X and Y]
        
    
    def collect_evidence(self, cluster_x,  caller=None):
        # mark X
        cluster_x[2] = True
        sepset_x_caller = None
        for sepset in self.sepsets:
            if cluster_x[0]==sepset[1]:
                if caller != None:
                    if caller[0]==sepset[2]:
                        sepset_x_caller = sepset
                neighbor = self.get_cluster_fac_mark(sepset[2])
                if neighbor[2] == False:
                    self.collect_evidence(neighbor, cluster_x)
            if cluster_x[0]==sepset[2]:
                if caller != None:
                    if caller[0]==sepset[1]:
                        sepset_x_caller = sepset
                neighbor = self.get_cluster_fac_mark(sepset[1])
                if neighbor[2] == False:
                    self.collect_evidence(neighbor, cluster_x)
            
        if caller != None:
            self.pass_message(cluster_x, sepset_x_caller ,caller)
                
        
    def distribute_evidence(self, cluster_x):
        # mark X
        cluster_x[2] = True    
        neighbors_unmarked = []
        
        for sepset in self.sepsets:
            if cluster_x[0]==sepset[1]:
                neighbor = self.get_cluster_fac_mark(sepset[2])
                if neighbor[2] == False:
                    self.pass_message(cluster_x, sepset , neighbor)
                    neighbors_unmarked.append(neighbor)
            if cluster_x[0]==sepset[2]:
                neighbor = self.get_cluster_fac_mark(sepset[1])
                if neighbor[2] == False:
                    self.pass_message(cluster_x, sepset , neighbor)
                    neighbors_unmarked.append(neighbor)
            
        for neighbor in neighbors_unmarked:
            self.distribute_evidence(neighbor)
            
            
    def set_to_zero(self, array_part, axis_left):
        assert axis_left > 0, "number of axis should be at least 1"
        if axis_left == 1:
            for i in range(len(array_part)):
                array_part[i] = 0
        else:
            for i in range(len(array_part)):
                self.set_to_zero(array_part[i], axis_left-1)