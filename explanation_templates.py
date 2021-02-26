
starcraft_actions = {
                    91: 'build supply depot',
                    42: 'build barracks',
                    477: 'train marine',
                    13: 'attack',
                    }

starcraft_features = {
                     0: 'workers',
                     1: 'supply depot',
                     2: 'barracks',
                     3: 'marines',
                     4: 'army health',
                     5: 'ally location',
                     6: 'enemy location',
                     7: 'destroyed units',
                     8: 'destroyed buildings',
                    }

def sc_generate_why_text_explanations(min_tuple_actual_state, min_tuple_optimal_state, action):
    exp_string = 'Because: goal is to increase' 
    for reward in min_tuple_actual_state['reward']:               
        exp_string += ', ' + str(starcraft_features[reward[0]])

    if len(min_tuple_actual_state['immediate']) > 1:
        exp_string += ': Which is influenced by'

        for immed in min_tuple_actual_state['immediate']:               
            exp_string += ', '+ str(starcraft_features[immed[0]]) +' (current '+str(immed[1])+')'
            for op_imm in min_tuple_optimal_state['immediate']:
                exp_string += ' (optimal '+str(op_imm[1])+') '

    if len(min_tuple_actual_state['head']) > 0:
        exp_string += ': that depend on'

        for immed in min_tuple_actual_state['head']:               
            exp_string += ', '+ str(starcraft_features[immed[0]]) +' (current '+str(immed[1])+')'
            for op_imm in min_tuple_optimal_state['head']:
                exp_string += ' (optimal '+str(op_imm[1])+') '

    return exp_string
     
def sc_generate_contrastive_text_explanations(minimal_tuple, action):
    exp_string = 'Because it is more desirable to do action ' + str(starcraft_actions[action]) + ', ' 

    for key in minimal_tuple['actual'].keys():
        if  minimal_tuple['actual'][key] >= minimal_tuple['counterfactual'][key]:
            exp_string += 'to have more ' + str(starcraft_features[key]) + ' (actual '+str(minimal_tuple['actual'][key])+') (counterfactual '+str(minimal_tuple['counterfactual'][key])+'), '
        if  minimal_tuple['actual'][key] < minimal_tuple['counterfactual'][key]:
            exp_string += 'to have less ' + str(starcraft_features[key]) + ' (actual '+str(minimal_tuple['actual'][key])+') (counterfactual '+str(minimal_tuple['counterfactual'][key])+'), '
    exp_string += 'as the goal is to have '
    for key in minimal_tuple['reward'].keys():               
        exp_string += '' + str(starcraft_features[key]) + ', '
    return exp_string