import tensorflow as tf
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import logging
import explanation_templates as explanations



#starcraft causal graph
graph_matrix = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0, 0], #0
    [0, 0, 0, 1, 0, 0, 0, 0, 0], #1
    [0, 0, 0, 1, 0, 0, 0, 0, 0], #2
    [0, 0, 0, 0, 0, 0, 0, 1, 1], #3
    [0, 0, 0, 0, 0, 0, 0, 1, 1], #4
    [0, 0, 0, 0, 0, 0, 0, 1, 1], #5
    [0, 0, 0, 0, 0, 0, 0, 1, 1], #6
    [0, 0, 0, 0, 0, 0, 0, 0, 0], #7
    [0, 0, 0, 0, 0, 0, 0, 0, 0], #8  
    ])


"""
action numbers and names, taken from actions.py in pysc2:

build_supply_depot = 91
build_barracks = 42
train_marine = 477
attack = 13

"""
actionset = (91, 42, 477, 13)
action_matrix = np.array([
    [0, 91, 42, 0, 0, 0, 0, 0, 0], #0
    [0, 0, 0, 477, 0, 0, 0, 0, 0], #1
    [0, 0, 0, 477, 0, 0, 0, 0, 0], #2
    [0, 0, 0, 0, 0, 0, 0, 13, 13], #3
    [0, 0, 0, 0, 0, 0, 0, 13, 13], #4
    [0, 0, 0, 0, 0, 0, 0, 13, 13], #5
    [0, 0, 0, 0, 0, 0, 0, 13, 13], #6
    [0, 0, 0, 0, 0, 0, 0, 0, 0], #7
    [0, 0, 0, 0, 0, 0, 0, 0, 0], #8  
    ])    

action_influence_dataset = {}
structeral_equations = {}
equation_predictions = {}

causal_graph = nx.from_numpy_matrix(graph_matrix, create_using=nx.MultiDiGraph())

for edge in causal_graph.edges():
    causal_graph.remove_edge(edge[0], edge[1])
    causal_graph.add_edge(edge[0], edge[1], action=action_matrix[edge[0]][edge[1]])


def train_structeral_equations():

    for key in structeral_equations:
        structeral_equations[key]['function'].train(input_fn=get_input_fn(structeral_equations[key],                                       
                                                                            num_epochs=None,                                      
                                                                            n_batch = 128,                                      
                                                                            shuffle=False),                                      
                                                                            steps=1000)    



def initialize_structeral_equations(config):

    uniqueu_functions = {}
    for edge in causal_graph.edges():
        predcs = causal_graph.predecessors(edge[1])        
        for pred in predcs:
            action_on_edge = action_matrix[pred][edge[1]]
            if (edge[1], action_on_edge) in uniqueu_functions:
                uniqueu_functions[(edge[1], action_on_edge)].add(pred)
            else:
                uniqueu_functions[(edge[1], action_on_edge)] = set()
                uniqueu_functions[(edge[1], action_on_edge)].add(pred)     
            

    for key in uniqueu_functions:
        if key[1] in action_influence_dataset:
            x_data = []
            for x_feature in uniqueu_functions[key]:
                x_data.append(np.array(action_influence_dataset[key[1]]['state'])[:,x_feature])

            x_feature_cols = [tf.feature_column.numeric_column(str(i)) for i in range(len(x_data))]  
            y_data = np.array(action_influence_dataset[key[1]]['next_state'])[:,key[0]]
            structeral_equations[key] = {
                                        'X': x_data,
                                        'Y': y_data,
                                        'function': get_regressor(x_feature_cols, key, config.scm_regressor)
                                        }


"""use different types of regressors"""        	
def get_regressor(x_feature_cols, key, regressor_type):
    if regressor_type == 'lr':
        return tf.estimator.LinearRegressor(feature_columns=x_feature_cols, model_dir='scm_models/linear_regressor/'+str(key[0])+'_'+str(key[1]))
    if regressor_type == 'mlp':
        return tf.estimator.DNNRegressor(hidden_units=[64, 32, 16], feature_columns=x_feature_cols, model_dir='scm_models/mlp/'+str(key[0])+'_'+str(key[1]))
    if regressor_type == 'dt':
        return tf.estimator.BoostedTreesRegressor(feature_columns=x_feature_cols, n_batches_per_layer=1000, n_trees=1, model_dir='scm_models/decision_tree/'+str(key[0])+'_'+str(key[1]))        

"""
get dictonary from expereince replay by making dictonary. first find the changed variables from S_i to S_i+1.  

for each changed feature s value: dict[tuple(action, s)] = array.append(S_i)
"""
def process_explanations(state_set, action_set, config, state_idx, agent_step, next_state_set=None):
    
    if (next_state_set==None):
        next_state_set = state_set[1:]
        action_set = action_set[:-1]
        state_set = state_set[:-1]

    if config.scm_mode == 'train':
        print('starting scm training-------')          
        for i in range(len(action_set)):

            if action_set[i] in action_influence_dataset:
                action_influence_dataset[action_set[i]]['state'].append(state_set[i])
                action_influence_dataset[action_set[i]]['next_state'].append(next_state_set[i])
            else:
                action_influence_dataset[action_set[i]] = {'state' : [], 'next_state': []}
                action_influence_dataset[action_set[i]]['state'].append(state_set[i])
                action_influence_dataset[action_set[i]]['next_state'].append(next_state_set[i])

        initialize_structeral_equations(config)
        train_structeral_equations()
        print('end scm training-------')
    else:
        uniqueu_actions = list(actionset)    
        for action in uniqueu_actions:
            action_influence_dataset[action] = {'state' : state_set, 'next_state': next_state_set}

        initialize_structeral_equations(config)

        why_explanations = {}
        why_not_explanations = {}
        for action in actionset:
            why_explanations[(agent_step, action)] = {'state': state_set[state_idx], 'why_exps': generate_why_explanations(state_set[state_idx], action, state_idx)}
            poss_counter_actions = set(actionset).difference({action})
            for counter_action in poss_counter_actions:
                why_not_explanations[(agent_step, action, counter_action)] = {'state': state_set[state_idx], 
                                                        'why_not_exps': generate_counterfactual_explanations(state_set[state_idx], action, counter_action, state_idx)}

        pd.DataFrame.from_dict(data=why_explanations, orient='index').to_csv('why_explanations.csv', mode='a', header=False)
        pd.DataFrame.from_dict(data=why_not_explanations, orient='index').to_csv('why_not_explanations.csv', mode='a', header=False)

    
    

def predict_from_scm():
    predict_y = {}
    for key in structeral_equations:
        training_pred = structeral_equations[key]['function'].predict(input_fn=get_input_fn(structeral_equations[key],                          
                num_epochs=1,                          
                n_batch = 128,                          
                shuffle=False))
        predict_y[key] = np.array([item['predictions'][0] for item in training_pred])
    equation_predictions = predict_y

def predict_node_scm (node, action):
    key = (node, action)
    pred = structeral_equations[key]['function'].predict(input_fn=get_input_fn(structeral_equations[key],                          
                num_epochs=1,                          
                n_batch = 128,                          
                shuffle=False))
    return  np.array([item['predictions'][0] for item in pred])           



def generate_why_explanations(actual_state, actual_action, state_num_in_batch):
    optimal_state_set = []
    actual_state = {k: actual_state[k] for k in range(len(actual_state))}
    sink_nodes = get_sink_nodes()
    actual_action_edge_list = get_edges_of_actions(actual_action)
    all_actual_causal_chains_from_action = get_causal_chains(sink_nodes, actual_action_edge_list)
    action_chain_list = get_action_chains(actual_action, all_actual_causal_chains_from_action)

    why_exps = set()
    for i in range(len(all_actual_causal_chains_from_action)):
        optimal_state = dict(actual_state)
        for j in range(len(all_actual_causal_chains_from_action[i])):
            for k in range(len(all_actual_causal_chains_from_action[i][j])):
                optimal_state[all_actual_causal_chains_from_action[i][j][k]] = predict_node_scm(
                    all_actual_causal_chains_from_action[i][j][k], action_chain_list[i][j][k])[state_num_in_batch]
        optimal_state_set.append(optimal_state)
        min_tuple_actual_state = get_minimally_complete_tuples(all_actual_causal_chains_from_action[i], actual_state)
        min_tuple_optimal_state = get_minimally_complete_tuples(all_actual_causal_chains_from_action[i], optimal_state)
        why_exps.add(explanations.sc_generate_why_text_explanations(min_tuple_actual_state, min_tuple_optimal_state, actual_action))

    return why_exps
 

def generate_counterfactual_explanations(actual_state, actual_action, counterfactual_action, state_num_in_batch):

    counterfactual_state_set = []
    actual_state = {k: actual_state[k] for k in range(len(actual_state))}
    sink_nodes = get_sink_nodes()
    counter_action_edge_list = get_edges_of_actions(counterfactual_action)
    actual_action_edge_list = get_edges_of_actions(actual_action)
    
    all_counter_causal_chains_from_action = get_causal_chains(sink_nodes, counter_action_edge_list)
    all_actual_causal_chains_from_action = get_causal_chains(sink_nodes, actual_action_edge_list)
    action_chain_list = get_action_chains(counterfactual_action, all_counter_causal_chains_from_action)
    
    for i in range(len(all_counter_causal_chains_from_action)):
        counterfactual_state = dict(actual_state)
        for j in range(len(all_counter_causal_chains_from_action[i])):
            for k in range(len(all_counter_causal_chains_from_action[i][j])):
                counterfactual_state[all_counter_causal_chains_from_action[i][j][k]] = predict_node_scm(
                    all_counter_causal_chains_from_action[i][j][k], action_chain_list[i][j][k])[state_num_in_batch]
        counterfactual_state_set.append(counterfactual_state)    
    
    contrastive_exp = set()
    for actual_chains in all_actual_causal_chains_from_action:
        for counter_chains in all_counter_causal_chains_from_action:
            for counter_states in counterfactual_state_set:
                contrast_tuple = get_minimal_contrastive_tuples(actual_chains, counter_chains, actual_state, counter_states)
                contrastive_exp.add(explanations.sc_generate_contrastive_text_explanations(contrast_tuple, actual_action))    
    #unuqieue contrastive explanations
    return contrastive_exp                 
    
    
"""minimally complete tuple = (head node of action, immediate pred of sink nodes, sink nodes)"""
def get_minimally_complete_tuples(chains, state):
    head = set()
    immediate = set()
    reward = set()
    for chain in chains:
        if len(chain) == 1:
            reward.add((chain[0], state[chain[0]]))
        if len(chain) == 2:
            head.add((chain[0], state[chain[0]]))
            reward.add((chain[-1], state[chain[-1]]))
        if len(chain) > 2:    
            head.add((chain[0], state[chain[0]]))
            immediate.add((chain[-2], state[chain[-2]]))
            reward.add((chain[-1], state[chain[-1]]))
    minimally_complete_tuple = {
        'head': head,
        'immediate': immediate,
        'reward': reward
    }
    return minimally_complete_tuple    

def get_minimal_contrastive_tuples(actual_chain, counterfactual_chain, actual_state, counterfactual_state):

    actual_minimally_complete_tuple = get_minimally_complete_tuples(actual_chain, actual_state)
    counterfactual_minimally_complete_tuple = get_minimally_complete_tuples(counterfactual_chain, counterfactual_state)
    min_tuples = np.sum(np.array([list(k) for k in list(actual_minimally_complete_tuple.values())]))
    tuple_states = set([k[0] for k in min_tuples])

    counter_min_tuples = np.sum(np.array([list(k) for k in list(counterfactual_minimally_complete_tuple.values())]))
    counter_tuple_states = set([k[0] for k in counter_min_tuples])
    counter_tuple_states.difference_update(tuple_states)

    contrastive_tuple = {
                        'actual': {k: actual_state[k] for k in counter_tuple_states},
                        'counterfactual': {k: counterfactual_state[k] for k in counter_tuple_states},
                        'reward': {k[0]: k[1] for k in actual_minimally_complete_tuple['reward']}
                        }
    return contrastive_tuple


def get_causal_chains(sink_nodes, action_edge_list):

    counter_action_head_set = set(np.array(action_edge_list)[:,1]) 
    all_causal_chains_from_action = []
    for action_head in counter_action_head_set:
        chains_to_sink_nodes = []
        for snode in sink_nodes:
            if action_head == snode:
                chains_to_sink_nodes.append([snode])
            else:
                chains_to_sink_nodes.extend((nx.all_simple_paths(causal_graph, source=action_head, target=snode)))
        all_causal_chains_from_action.append(chains_to_sink_nodes)
    return all_causal_chains_from_action    


def get_action_chains(action, chain_lists_of_action):
    action_chain_list = []
    for chain_list in chain_lists_of_action:
        action_chains = []
        for chain in chain_list:
            action_chain = []
            for i in range(len(chain)):
                if i == 0:
                    action_chain.append(action)  
                else:
                    action_chain.append(causal_graph.get_edge_data(chain[i-1], chain[i])[0]['action'])
            action_chains.append(action_chain)
        action_chain_list.append(action_chains)          
    return action_chain_list        


def get_edges_of_actions(action):
    return list(edge for edge in causal_graph.edges(data=True) if edge[2]['action'] == action)
   
def get_sink_nodes():
    return list((node for node, out_degree in causal_graph.out_degree_iter() if out_degree == 0 and causal_graph.in_degree(node) > 0 ))

def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=False):
        x_data = {str(k): data_set['X'][k] for k in range(len(data_set['X']))}
        return tf.estimator.inputs.pandas_input_fn(       
                x=pd.DataFrame(x_data),
                y = pd.Series(data_set['Y']),       
                batch_size=n_batch,          
                num_epochs=num_epochs,       
                shuffle=shuffle)

