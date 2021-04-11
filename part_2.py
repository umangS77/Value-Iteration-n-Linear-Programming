from collections import OrderedDict
from typing import Iterable
import os
import json
import random
import numpy as np

GAMMA = 0.999
DELTA = 1e-3

# TEAM 38 - YogiJi

arr = [1/2, 1, 2]
y = arr[38%3]

STEP_COST = -10/y
MOVEMENT_PROBABILITY = 0.85
HIT_PROBABILITY = 0.75
SHOOT_PROBABILITY = 0.25
# HEALING_POINTS = 25
MONSTER_STATE_CHANGE_PROB = 0.2
ATTACK_MONSTER_PROBABILITY = 0.5
MONSTER_DAMAGE = 25
ARROW_HIT_PROB_E = 0.9
ARROW_HIT_PROB_C = 0.5
ARROW_HIT_PROB_W = 0.25
ONE_ARROW_HIT_PROB = 0.5
TWO_ARROW_HIT_PROB = 0.35
THREE_ARROW_HIT_PROB = 0.15
MONSTER_KILL_REWARD = 50
IJ_HIT_PENALTY = 40


class DotDict(OrderedDict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self) -> Iterable[str]:
        l = list(dict.keys(self))
        l.extend(super(DotDict, self).__dir__())
        return l


class npdict():
    def __init__(self, nparr, one_one_mapper):
        self.arr = nparr
        self.mapper = one_one_mapper

    def __getitem__(self, key):
        return self.arr[self.mapper[key]]

    def __setitem__(self, key, val):
        self.arr[self.mapper[key]] = val


positions = {
    (0,0): "C",
    (-1,0): "W",
    (1,0): "E",
    (0,1): "N",
    (0,-1): "S"
}

coordinates = {
    "C": (0,0),
    "W": (-1,0),
    "E": (1,0),
    "N": (0,1),
    "S": (0,-1)
}

actions_list = {
    "C": ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "SHOOT", "HIT"],
    "W": ["RIGHT", "STAY", "SHOOT"],
    "E": ["LEFT", "STAY", "SHOOT", "HIT"],
    "N": ["DOWN", "STAY", "CRAFT"],
    "S": ["UP", "STAY", "GATHER"]
}

# eg. the first element of a state tuple is 1 -> state corresponds to directions_list[1] = "W" current_pos

directions_list = ["C", "W", "E", "N", "S"]
codes_loc = ["C", "W", "E", "N", "S"]
codes_state = ["D", "R"]


output = []







# MOVEMENT FUNCTIONS

# def move_up(state):
#     temp_state = list(state.copy())
#     old_index = coordinates[directions_list[temp_state[0]]]
#     (x, y) = old_index
#     y += 1    # increment one direction along y-axis
#     # coord = (x, y)
#     temp_state[0] = directions_list.index(positions[(x, y)])
#     return temp_state
    

# def move_down(state):
#     temp_state = list(state.copy())
#     old_index = coordinates[directions_list[temp_state[0]]]
#     (x, y) = old_index
#     y -= 1
#     # coord = (x, y)
#     temp_state[0] = directions_list.index(positions[(x, y)])
#     return temp_state
    

# def move_left(state):
#     temp_state = list(state.copy())
#     old_index = coordinates[directions_list[temp_state[0]]]
#     (x, y) = old_index
#     x -= 1
#     # coord = (x, y)
#     temp_state[0] = directions_list.index(positions[(x, y)])
#     return temp_state
    

# def move_right(state):
#     temp_state = list(state.copy())
    
#     old_index = coordinates[directions_list[temp_state[0]]]
#     (x, y) = old_index
#     x += 1
#     # coord = (x, y)
#     temp_state[0] = directions_list.index(positions[(x, y)])
#     return temp_state
    

# def move_stay(state):
#     temp_state = list(state.copy())
#     return temp_state
    

def move(action, state):
    # switcher = {
    #     "UP": move_up,
    #     "DOWN": move_down,
    #     "LEFT": move_left,
    #     "RIGHT": move_right,
    #     "STAY": move_stay
    # }
    temp_state = list(state.copy())
    old_index = coordinates[directions_list[temp_state[0]]]
    (x, y) = old_index

    if action == "UP":
        y += 1
    elif action == "DOWN":
        y -= 1
    elif action == "LEFT":
        x -= 1
    elif action == "RIGHT":
        x += 1
    elif action == "STAY":
        pass

    temp_state[0] = directions_list.index(positions[(x, y)])    
    return(temp_state)



def probability_calculation(state, action):
    current_pos = directions_list[state[0]]
    choice = []
    
    if current_pos == "N":
        if action in ["DOWN", "STAY"]:
            temp_state = state.copy()
            curr_state = state.copy()
            next_state = move(action, temp_state)
            curr_state[0] = 2
            choice = [[MOVEMENT_PROBABILITY, 1 - MOVEMENT_PROBABILITY], [next_state, curr_state]]
        else:
            arrows_hit_probabilities = [ONE_ARROW_HIT_PROB, TWO_ARROW_HIT_PROB, THREE_ARROW_HIT_PROB]
            a1 = state.copy()
            a1[1] -= 1
            a1[2] = min(3, a1[2] + 1)

            a2 = state.copy()
            a2[1] -= 1
            a2[2] = min(3, a2[2] + 2)
            
            a3 = state.copy()
            a3[1] -= 1
            a3[2] = min(3, a3[2] + 3)
            
            choice = [arrows_hit_probabilities, [a1, a2, a3]]
            
  
    elif current_pos == "S":
        if action in ["UP", "STAY"]:
            temp_state = state.copy()
            curr_state = state.copy()
            next_state = move(action, temp_state)
            curr_state[0] = 2
            choice = [[MOVEMENT_PROBABILITY, 1 - MOVEMENT_PROBABILITY], [next_state, curr_state]]
        else:
            gather_probs = [0.75, 0.25]
            next_state = state.copy()
            curr_state = next_state.copy()
            next_state[1] = min(2, next_state[1] + 1)
            choice = [gather_probs, [next_state, curr_state]]
                
                
    elif current_pos == "W":
        if action in ["RIGHT", "STAY"]:
            temp_state = state.copy()
            curr_state = state.copy()
            next_state = move(action, temp_state)
            curr_state[0] = 2
            choice = [[1, 0], [next_state, curr_state]]
        else:
            shoot_probs = [ARROW_HIT_PROB_W, 1 - ARROW_HIT_PROB_W]
            next_state = state.copy()
            next_state[2] -= 1
            curr_state = next_state.copy()
            next_state[4] -= 1
            choice = [shoot_probs, [next_state, curr_state]]
            
            
    elif current_pos == "E":
        if action in ["LEFT", "STAY"]:
            temp_state = state.copy()
            curr_state = state.copy()
            next_state = move(action, temp_state)
            curr_state[0] = 2
            choice = [[1, 0], [next_state, curr_state]]
        elif action == "SHOOT":
            shoot_probs = [ARROW_HIT_PROB_E, 1 - ARROW_HIT_PROB_E]
            next_state = state.copy()
            next_state[2] -= 1
            curr_state = next_state.copy()
            next_state[4] -= 1
            choice = [shoot_probs, [next_state, curr_state]]
        else:
            hit_probs = [0.2, 0.8]
            next_state = state.copy()
            curr_state = next_state.copy()
            next_state[4] = max(0, next_state[4] - 2)
            choice = [hit_probs, [next_state, curr_state]]
            
            
    elif current_pos == "C":
        if action in ["UP", "DOWN", "RIGHT", "LEFT", "STAY"]:
            temp_state = state.copy()
            curr_state = state.copy()
            next_state = move(action, temp_state)
            curr_state[0] = 2
            choice = [[MOVEMENT_PROBABILITY, 1 - MOVEMENT_PROBABILITY], [next_state, curr_state]]
        elif action == "SHOOT":
            shoot_probs = [ARROW_HIT_PROB_C, 1 - ARROW_HIT_PROB_C]
            next_state = state.copy()
            next_state[2] -= 1
            curr_state = next_state.copy()
            next_state[4] -= 1
            choice = [shoot_probs, [next_state, curr_state]]
        else:
            hit_probs = [0.1, 0.9]
            next_state = state.copy()
            curr_state = next_state.copy()
            next_state[4] = max(0, next_state[4] - 2)
            choice = [hit_probs, [next_state, curr_state]]
            
    # monster slep slep 
    if not state[3]:
        awake_prob = [0.2, 0.8]
        probabilities = choice[0]
        states = choice[1]
        
        new_probs = []
        new_states = []
        
        for i, prob in enumerate(probabilities):
            new_probs.append(prob*awake_prob[0])
            new_probs.append(prob*awake_prob[1])
            awake_state = states[i].copy()
            awake_state[3] = 1
            new_states.append(awake_state)
            new_states.append(states[i].copy())
        
        choice = [new_probs, new_states]

    # monster wakey wakey   
    else:
        probabilities = choice[0]
        states = choice[1]
        
        new_probs = []
        new_states = []
        
        # if not current_pos == "E" or "C":
        attacc_prob = [0.5, 0.5]
        
        for i, prob in enumerate(probabilities):
            new_probs.append(prob*attacc_prob[0])
            new_states.append(states[i].copy())
        # new_probs.append(attacc_prob[1])
        
        
        

        for i, prob in enumerate(probabilities):
            awake_state = states[i].copy()

            if current_pos == "E" or "C":
                og_state = state.copy()
                og_state[2] = 0 # arrows
                og_state[4] = min(4, og_state[4] + 1) # helth
            
            og_state[3] = 0 # dormant state
                            
            new_probs.append(prob*attacc_prob[0])
            new_states.append(og_state.copy())
        choice = [new_probs, new_states]
            
    return choice


# calculate rewards
# def reward(old, new):
#     reward = STEP_COST
    
#     # monster dies
#     if new[4] == 0:
#         reward += 50
        
#     # monster attaccs indiana
#     elif directions_list[old[0]] in ["C", "E"] and old[3] == 1 and new[3] == 0:
#         reward -= 40
        
#     return reward

# print trace
def send_output(state, chosen_action, biggest_utility):
    trace_state = state.copy()
    trace_state[0] = codes_loc[trace_state[0]]
    trace_state[3] = codes_state[trace_state[3]]
    trace_state[4] = trace_state[4]*MONSTER_DAMAGE

    # trace_action = chosen_action
    print_utility = '{:.3f}'.format(np.round(biggest_utility, 3))
    
    output_string = "(" + ",".join([str(i) for i in trace_state]) + "):" + chosen_action + "=[" + print_utility + "]"
        
    # to_print += "("
    # trace_state = ",".join([str(i) for i in trace_state])
    # to_print += trace_state + "):"
    # to_print += trace_action + "=["
    # to_print += trace_utility + "]"

    print(output_string)




## iteration loop

def main():
    old_utilities = []
    error = 1e10
    utilities = np.zeros((5,3,4,2,5))
    old_utilities.append(utilities)

    iteration_index = 1
    while(error > DELTA):
        this_iter_actions = np.full((5,3,4,2,5), "temp")
        this_utilities = np.zeros((5,3,4,2,5))

        print("iteration=" + str(iteration_index))
        for state, x in np.ndenumerate(utilities):
            
            state = list(state)
            util_per_act = []
            actions_now = actions_list[directions_list[state[0]]].copy()

            if 'SHOOT' in actions_now:
                if state[2] == 0:
                    actions_now.remove('SHOOT')

            if 'CRAFT' in actions_now:
                if state[1] == 0:
                    actions_now.remove('CRAFT')

            if state[4] == 0:
                actions_now = ['NONE']
                        
            for a in actions_now:
                utility = old_utilities[-1][tuple(state)]        
                
                if a == 'NONE':
                    pass
                else:
                    utility = 0
                    p, s = probability_calculation(state, a)
                    
                    tp = 0

                    for i, pp in enumerate(p):
                        this_state = s[i].copy()
                        tp = GAMMA*(old_utilities[-1][tuple(this_state)])
                        new_reward = STEP_COST
                        if this_state[4] <= 0:
                            new_reward = STEP_COST + MONSTER_KILL_REWARD
                        elif directions_list[state[0]] in ["C", "E"]:
                            if state[3] == 1:
                                if this_state[3] == 0:
                                    new_reward = STEP_COST - IJ_HIT_PENALTY
                        utility = utility + pp*(new_reward + tp)

                util_per_act.append(utility)
                
            biggest_utility = max(util_per_act)
            index_of_big_ut = len(util_per_act) - 1 - util_per_act[::-1].index(biggest_utility)



            # index_of_big_ut = util_per_act.index(biggest_utility)
            chosen_action = actions_now[index_of_big_ut]
            this_utilities[tuple(state)] = biggest_utility

            # print(biggest_utility)
            send_output(state.copy(), chosen_action, biggest_utility)


            this_iter_actions[tuple(state)] = chosen_action

            
        old_utilities.append(this_utilities)
        
        error = np.max(np.abs(old_utilities[-1] - old_utilities[-2]))
        
        output = this_iter_actions

        iteration_index += 1




    # data for simulation
    # output = np.array(output)
    # output = output.tolist()
    # with open('output.json', 'w') as f:
    #     json.dump(output, f)


main()