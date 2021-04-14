import numpy as np
import cvxpy as cp
import os
import json
import random

GAMMA = 0.999
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


states = []

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

def main():
    temp = []
    iteration_index = 0
    print_vals = dict()

    st_to_id = np.zeros((5, 3, 4, 2, 5))

    actions_list = {"C": ["DOWN", "HIT", "LEFT", "RIGHT", "SHOOT", "STAY", "UP"],
                "W": ["RIGHT", "SHOOT", "STAY"],
                "E": ["HIT", "LEFT", "SHOOT", "STAY"],
                "N": ["CRAFT", "DOWN", "STAY"],
                "S": ["UP", "STAY", "GATHER"],}

    for stx, y in np.ndenumerate(st_to_id):
        
        actions_now = actions_list[directions_list[stx[0]]].copy()        
        st_to_id[stx] = iteration_index
        states.append(stx)

        if "SHOOT" in actions_now and stx[2] == 0:
                actions_now.remove("SHOOT")

        if "CRAFT" in actions_now and stx[1] == 0:
                actions_now.remove("CRAFT")
        
        if stx[4] == 0:
            actions_now = ["NONE"]
        
        iteration_index = iteration_index + 1

        for z in actions_now:
            temp.append((stx, z))
    
    st_acn_p = temp
    
    final_A = np.zeros((len(states),len(st_acn_p)))
    final_R = np.zeros(len(st_acn_p))

    for i, x_action in enumerate(st_acn_p):

        if x_action[1] == "NONE":
            final_A[int(st_to_id[x_action[0]])][i] = 1
            

        else:
            tmp_acn_lst = list(x_action[0])
            v, stx = probability_calculation(tmp_acn_lst, x_action[1])
            yy = tuple(x_action[0])
            final_A[int(st_to_id[yy])][i] += np.sum(v)

            for j, p in enumerate(v):
                l = tuple(stx[j])
                final_A[int(st_to_id[l])][i] -= p


    for k , act in enumerate(st_acn_p):

        if act[1] == "NONE":
            pass
        else:
            v, stx = probability_calculation(list(act[0]), act[1])
            
            for j, p in enumerate(v):

                cost = STEP_COST
                if (act[0][3] == 1):
                    if (stx[j][3]==0):
                        if (act[0][0] in [0,2]):
                            if (act[0][4]>0):
                                cost -= IJ_HIT_PENALTY

                final_R[k] += p*cost
    
    alpha = np.full((len(states),1),1/600)
    x = cp.Variable(shape=(len(st_acn_p),1), name="x")
    
    cons = [cp.matmul(final_A, x) == alpha, x >= 0]

    obj = cp.Maximize(cp.matmul(final_R,x))
    
    state_symb = ["C", "W", "E", "N", "S"]
    
    prob = cp.Problem(obj, cons)

    answer = x, prob.solve()

    print_vals["a"] = [ list(i) for i in final_A ]
    print_vals["r"] = list(final_R)

    output_policy = []
    
    print_vals["alpha"] = [ i[0] for i in alpha]
    
    codes_state = ["D", "R"]

    print_vals["x"] = [i[0] for i in answer[0].value]

    for s in states:
        act = ""
        highest_rew = -9999999999 

        for j, pair in enumerate(st_acn_p):
            r = answer[0].value[j][0]
            if s == pair[0] and highest_rew < r:
                    highest_rew = max(highest_rew,r)
                    act = pair[1]
        lst = list(s).copy()
        lst[3] = codes_state[s[3]]
        lst[4] = MONSTER_DAMAGE*lst[4]
        lst[0] =  state_symb[s[0]]

        tmp_out = ([tuple(lst),act])
        output_policy.append(tmp_out)


    
    print_vals["policy"] = output_policy
    print_vals["objective"] = answer[1]

    with open('./outputs/part_3_output.json','w') as f:
        json.dump(print_vals,f)

main()