from collections import OrderedDict
from typing import Iterable
import os
import json
import random
import numpy as np

GAMMA = 0.999

# task 2.3
# GAMMA = 0.25

DELTA = 1e-3
FL = 2

# TEAM 38 - YogiJi

arr = [1/2, 1, 2]
Y = arr[38%3]
STEP_COST = -10/Y
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
GATHER_PROB = 0.75
ATTACK_PROB = 0.5
MONS_AWAKE_PROB = 0.2
BLADE_HIT_PROB_E = 0.2
BLADE_HIT_PROB_C = 0.1


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


directions_list = ["C", "W", "E", "N", "S"]

def move(action, state):
    temp_state = list(state.copy())
    indexes = {
    "N": (0,1),
    "S": (0,-1),
    "C": (0,0),
    "W": (-1,0),
    "E": (1,0),
    }
    old_index = indexes[directions_list[temp_state[0]]]
    (x, y) = old_index

    if action == "UP":
        y += 1
    elif action == "DOWN":
        y -= 1
    elif action == "LEFT":
        x -= 1
        
        # task 2.1
        # if old_index == indexes["E"]:
        #     (x, y) = indexes["W"]

    elif action == "RIGHT":
        x += 1
    elif action == "STAY":
        pass
    direcs = {
        (0,1): "N",
        (0,-1): "S",
        (0,0): "C",
        (-1,0): "W",
        (1,0): "E",
    }
    temp_state[0] = directions_list.index(direcs[(x, y)])    
    return(temp_state)

monster_state = {
    "D" : 0,
    "R" : 1
}


def probability_calculation(state, action):
    current_pos = directions_list[state[0]]
    choice = []
    
    if current_pos == "S":
        if action not in ["UP", "STAY"]:
            temp_state = state.copy()
            temp_state[1] += 1
            if temp_state[1] > 2:
                temp_state[1] = 2
            choice = [[GATHER_PROB, 1 - GATHER_PROB], [temp_state, state.copy()]]
        else:
            temp_state = state.copy()
            curr_state = state.copy()
            curr_state[0] = 2
            next_state = move(action, temp_state)
            choice = [[MOVEMENT_PROBABILITY, 1 - MOVEMENT_PROBABILITY], [next_state, curr_state]]

    elif current_pos == "N":
        if action not in ["DOWN", "STAY"]:
            a1 = state.copy()
            a1[1] -= 1
            a1[2] = min(3, a1[2] + 1)

            a2 = state.copy()
            a2[1] -= 1
            a2[2] = min(3, a2[2] + 2)
            
            a3 = state.copy()
            a3[1] -= 1
            a3[2] = min(3, a3[2] + 3)
            fin = [a1, a2, a3]
            choice = [[ONE_ARROW_HIT_PROB, TWO_ARROW_HIT_PROB, THREE_ARROW_HIT_PROB], fin]
        else:
            temp_state = state.copy()
            curr_state = state.copy()
            curr_state[0] = 2
            next_state = move(action, temp_state)
            choice = [[MOVEMENT_PROBABILITY, 1 - MOVEMENT_PROBABILITY], [next_state, curr_state]]
                
    elif current_pos == "W":
        if action not in ["RIGHT", "STAY"]:
            temp_state = state.copy()
            temp_state[2] -= 1
            curr_state = temp_state.copy()
            temp_state[4] -= 1
            choice = [[ARROW_HIT_PROB_W, 1 - ARROW_HIT_PROB_W], [temp_state, curr_state]]
        else:
            curr_state = state.copy()
            curr_state[0] = 2
            temp_state = state.copy()
            next_state = move(action, temp_state)
            choice = [[1, 0], [next_state, curr_state]]
            
            
    elif current_pos == "E":
        
        if action == "SHOOT":
            temp_state = state.copy()
            temp_state[2] = temp_state[2] - 1
            curr_state = temp_state.copy()
            temp_state[4] = temp_state[4] - 1
            choice = [[ARROW_HIT_PROB_E, 1 - ARROW_HIT_PROB_E], [temp_state, curr_state]]
        elif action not in ["LEFT", "STAY"]:
            temp_state = state.copy()
            curr_state = temp_state.copy()
            temp_state[4] = temp_state[4] - 2
            if temp_state[4] <= 0:
                temp_state[4] = 0
            choice = [[BLADE_HIT_PROB_E, 1 - BLADE_HIT_PROB_E], [temp_state, curr_state]]
        else:
            curr_state = state.copy()
            curr_state[0] = 2
            temp_state = state.copy()
            next_state = move(action, temp_state)
            choice = [[1, 0], [next_state, curr_state]]
            
            
    elif current_pos == "C":
        if action == "SHOOT":
            temp_state = state.copy()
            curr_state = temp_state.copy()
            temp_state[2] = temp_state[2] - 1
            curr_state = temp_state.copy()
            temp_state[4] = temp_state[4] - 1
            choice = [[ARROW_HIT_PROB_C, 1 - ARROW_HIT_PROB_C], [temp_state, curr_state]]
        elif action not in ["UP", "DOWN", "RIGHT", "LEFT", "STAY"]:
            temp_state = state.copy()
            curr_state = state.copy()
            temp_state[4] -= 2
            if temp_state[4] < 0:
                temp_state[4] = 0
            choice = [[BLADE_HIT_PROB_C, 1 - BLADE_HIT_PROB_C], [temp_state, curr_state]]

        else:
            temp_state = state.copy()
            curr_state = state.copy()
            next_state = move(action, temp_state)
            curr_state[0] = 2
            choice = [[MOVEMENT_PROBABILITY, 1 - MOVEMENT_PROBABILITY], [next_state, curr_state]]
            
            
    if state[3] == monster_state["R"]:
        probs = []
        temp_states = []
        states = choice[1]
        
        for i, p in enumerate(choice[0]):
            probs.append(p*ATTACK_PROB)
            temp_states.append(states[i].copy())
        for j, p in enumerate(choice[0]):
            mons_active = states[j].copy()
            og_state = state.copy()

            if current_pos == "E" or current_pos == "C":
                og_state[4] = min(4, og_state[4] + 1)
                og_state[2] = 0
                og_state[3] = monster_state["D"]
            probs.append(p*ATTACK_PROB)
            temp_states.append(og_state.copy())
    else:
        states = choice[1]
        probs = []
        temp_states = []
        for i, p in enumerate(choice[0]):
            mons_active = states[i].copy()
            probs.append(p*MONS_AWAKE_PROB)
            probs.append(p*(1 - MONS_AWAKE_PROB))
            mons_active[3] = monster_state["R"]
            temp_states.append(mons_active)
            temp_states.append(states[i].copy())
        
    return [probs, temp_states]

actions_list = {"C": ["DOWN", "HIT", "LEFT", "RIGHT", "SHOOT", "STAY", "UP"],
                "W": ["RIGHT", "SHOOT", "STAY"],
                "E": ["HIT", "LEFT", "SHOOT", "STAY"],
                "N": ["CRAFT", "DOWN", "STAY"],
                "S": ["UP", "STAY", "GATHER"],}

def main():
    old_utilities = []
    error = 1e10
    utilities = np.zeros((5,3,4,2,5))
    old_utilities.append(utilities)
    output = []
    iteration_index = 0
    file = open('./outputs/part_2_trace.txt', 'w')
    
    while(error > DELTA):
        this_iter_actions = np.full((5,3,4,2,5), "temp")
        final_util = np.zeros((5,3,4,2,5))

        file.write("iteration=" + str(iteration_index) + "\n")
        iteration_index += 1
        for state, x in np.ndenumerate(utilities):
            
            state = list(state)
            output_state = state.copy()
            actions_now = actions_list[directions_list[state[0]]].copy()

            if 'SHOOT' in actions_now:
                if state[2] == 0:
                    actions_now.remove('SHOOT')

            if 'CRAFT' in actions_now:
                if state[1] == 0:
                    actions_now.remove('CRAFT')

            util_per_act = []

            if state[4] == 0:
                actions_now = ['NONE']
                        
            for a in actions_now:
                util = old_utilities[-1][tuple(state)]        
                
                if a == 'NONE':
                    pass
                else:
                    util = 0
                    p, s = probability_calculation(state, a)
                    
                    tp = 0

                    for i, pp in enumerate(p):
                        this_state = s[i].copy()
                        tp = GAMMA*(old_utilities[-1][tuple(this_state)])
                        new_reward = STEP_COST

                        # task 2.2
                        # if a == 'STAY':
                        #     new_reward = 0


                        if this_state[4] <= 0:
                            new_reward = new_reward + MONSTER_KILL_REWARD
                        elif directions_list[state[0]] in ["C", "E"]:
                            if state[3] == monster_state["R"]:
                                if this_state[3] == 0:
                                    new_reward = new_reward - IJ_HIT_PENALTY
                        util = util + pp*(new_reward + tp)

                util_per_act.append(util)
            biggest_utility = max(util_per_act)
            final_util[tuple(state)] = biggest_utility
            chosen_action = actions_now[(len(util_per_act) - 1 - util_per_act[::-1].index(biggest_utility))]
            output_state = state.copy()
            codes_loc = ["C", "W", "E", "N", "S"]
            output_state[0] = codes_loc[output_state[0]]
            if output_state[3] == 0:
                output_state[3] = "D"
            else:
                output_state[3] = "R"
            output_state[4] = output_state[4]*MONSTER_DAMAGE
            output_string = "(" + ",".join([str(i) for i in output_state]) + "):" + chosen_action + "=[" + '{:.3f}'.format(np.round(biggest_utility, 3)) + "]\n"
            file.write(output_string)


            this_iter_actions[tuple(state)] = chosen_action

        
        old_utilities.append(final_util)
        output = this_iter_actions
        error = np.max(np.abs(old_utilities[-1] - old_utilities[-2]))

    file.close()


main()