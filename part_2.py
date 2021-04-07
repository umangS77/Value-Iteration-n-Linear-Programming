from collections import OrderedDict
from typing import Iterable
import os
import random
import numpy as np
# from dict import DotDict, npdict
GAMMA = 0.999
DELTA = 1e-3
FL = 2
# TEAM 44 - YogiJi
Y = [1/2, 1, 2]
STEP_COST = -5
MOVEMENT_PROBABILITY = 0.85
HIT_PROBABILITY = 0.75
HEALING_POINTS = 25
MONSTER_STATE_CHANGE_PROB = 0.2
ATTACK_MONSTER_PROBABILITY = 0.5
MONSTER_DAMAGE = 25
ARROW_HIT_PROB_E = 0.9
ARROW_HIT_PROB_C = 0.5
ARROW_HIT_PROB_W = 0.25
PROB_1_ARROW = 0.5
PROB_2_ARROW = 0.35
PROB_3_ARROW = 0.15
BLADE_HIT_PROB_E = 0.2
BLADE_HIT_PROB_C = 0.1
#       W    N     C    S    E
#   L
#   U
#   S
#   D
#   R
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



PlayerAttacks = DotDict({
    "SHOOT": 25,
    "HIT": 50,
})

PlayerActions = DotDict({
    "CRAFT": 2,
    "GATHER": 3,
    "NONE": 4,
})


class PlayerTakeStep:
    UP = np.array((0, 1))
    DOWN = np.array((0, -1))
    LEFT = np.array((-1, 0))
    RIGHT = np.array((1, 0))
    STAY = np.array((0, 0))


PlayerMoves = DotDict({
    "LEFT": tuple(PlayerTakeStep.LEFT),
    "UP": tuple(PlayerTakeStep.UP),
    "STAY": tuple(PlayerTakeStep.STAY),
    "DOWN": tuple(PlayerTakeStep.DOWN),
    "RIGHT": tuple(PlayerTakeStep.RIGHT),
})

class StateOfPlayer:
    W = np.array((-1, 0))
    N = np.array((0, 1))
    C = np.array((0, 0))
    S = np.array((0, -1))
    E = np.array((1, 0))

PlayerStates = DotDict({
    "W": tuple(StateOfPlayer.W),
    "N": tuple(StateOfPlayer.N),
    "C": tuple(StateOfPlayer.C),
    "S": tuple(StateOfPlayer.S),
    "E": tuple(StateOfPlayer.E),
})


_INV_STATE_MAP = {PlayerStates[k]: k for k in PlayerStates}
_INV_STATE_MAP_INDEX = {k: i for i, k in enumerate(PlayerStates)}
_INV_ACTIONS_MAP = {PlayerActions[k]: k for k in PlayerActions}
_INV_ACTIONS_MAP.update({PlayerAttacks[k]: k for k in PlayerAttacks})
_INV_ACTIONS_MAP.update({PlayerMoves[k]: k for k in PlayerMoves})

class IndexOfState:
    # W = 0
    # N = 1
    # C = 2
    # S = 3
    # E = 4
    _inv_map = DotDict({
        tuple(StateOfPlayer.W): 0,
        tuple(StateOfPlayer.N): 1,
        tuple(StateOfPlayer.C): 2,
        tuple(StateOfPlayer.S): 3,
        tuple(StateOfPlayer.E): 4,
    })

    # @staticmethod
    def source_state(state):
        return IndexOfState._inv_map[tuple(state)]

def probabilityTF(prob: float):
    tp = random.random()
    if tp >= prob:
        return False
    else:
        return True

class MightyMonster(object):
    STATES = DotDict({"D": 0, "R": 1})
    def __init__(self):
        self.health = 100
        self.state = MightyMonster.STATES.D

    def monsterAttacked(self, damage: int = MONSTER_DAMAGE):
        self.health -= damage

    def monster_attack_ij(self, player):
        if probabilityTF(ATTACK_MONSTER_PROBABILITY):
            # attackMonster
            if player.monster_attack_successful() == True:
                print("Monster attacked")
                self.health += HEALING_POINTS  # healing of monster
                self.health = min(self.health, 100)
                self.state = MightyMonster.STATES.D  # change monster state to dormant

    def take_step(self, player):
        if probabilityTF(MONSTER_STATE_CHANGE_PROB):
            self.state = MightyMonster.STATES.R  # change monster state to ready
        if self.state == MightyMonster.STATES.R:
            self.monster_attack_ij(player)

class IJones(object):
    
    ATTACKS = PlayerAttacks
    ACTIONS = PlayerActions
    MOVES = PlayerMoves
    STATES = PlayerStates

    def __init__(self):
        # self.name = "Indiana Jones"
        # ATTACKS = PlayerAttacks
        # ACTIONS = PlayerActions
        # MOVES = PlayerMoves
        # STATES = PlayerStates
        self.state = StateOfPlayer.N
        self._action = IJones.ACTIONS.NONE
        self.ij_step_flag = False
        self._values = np.zeros((5,), dtype=np.float64)
        self.arrows = 0
        self.materials = 0
        self.reward = 0

    def current_state(self) -> str:
        return _INV_STATE_MAP[tuple(self.state)]

    def action(self) -> str:
        return _INV_ACTIONS_MAP[self._action]

    def move_to_E(self):
        self.state = StateOfPlayer.E

    def move(self, d):
        self.state += d
        if np.sum(np.abs(self.state)) > 1:
            self.state -= d

    def execute_step(self, d: tuple):
        if not probabilityTF(FL):
            return
        if tuple(self.state) in [IJones.STATES.C, IJones.STATES.N, IJones.STATES.S]:
            if probabilityTF(MOVEMENT_PROBABILITY):
                pass
            else:
                self.move_to_E()
                return
        self.move(d)

    def take_step(self, monster):
        # self.value_iteration()
        # VALUE ITERATION
        val1 = np.zeros((5,), dtype=np.float64)
        val2 = np.zeros((5,), dtype=np.float64)
        state_index = IndexOfState.source_state(self.state)
        val1[0] = np.sum(player_board[state_index][0] * STEP_COST)
        val2[0] = GAMMA * np.sum(self._values.copy() * player_board[state_index][0])
        val1[1] = np.sum(player_board[state_index][1] * STEP_COST)
        val2[1] = GAMMA * np.sum(self._values.copy() * player_board[state_index][1])
        val1[2] = np.sum(player_board[state_index][2] * STEP_COST)
        val2[2] = GAMMA * np.sum(self._values.copy() * player_board[state_index][2])
        val1[3] = np.sum(player_board[state_index][3] * STEP_COST)
        val2[3] = GAMMA * np.sum(self._values.copy() * player_board[state_index][3])
        val1[4] = np.sum(player_board[state_index][4] * STEP_COST)
        val2[4] = GAMMA * np.sum(self._values.copy() * player_board[state_index][4])
        # lst = val1 + val2
        # maxcv = np.max(val1 + val2)
        self._values[state_index] = np.max(val1 + val2)
        # self._values[:] = np.round(self._values, 9)
        if self.ij_step_flag == True:
            self.ij_step_flag = False
            return
        if hasattr(self, 'choices'):
            pass
        else:
            # all possible choices player could make
            self.choices = []
            self.choices.extend(list(IJones.ATTACKS.values()))
            self.choices.extend(list(IJones.MOVES.values()))
            self.choices.extend(list(IJones.ACTIONS.values()))
        # perform a random action
        self._action = random.choice(self.choices)
        # self.perform_action(monster)
        # action = self._action
        if self._action in IJones.ATTACKS.values():
            self.attackMonster(monster, self._action)
        elif self._action in IJones.MOVES.values():
            self.execute_step(random.choice(list(IJones.MOVES.values())))
        elif self._action in IJones.ACTIONS.values():
            if self._action == IJones.ACTIONS.CRAFT:
                self.craft()
            elif self._action == IJones.ACTIONS.GATHER:
                self.gather()
        return

    @property
    def values(self):
        return npdict(self._values, _INV_STATE_MAP_INDEX)

    def monster_attack_successful(self):
        tp_flag = False
        if tuple(self.state) not in [IJones.STATES.C, IJones.STATES.E]:
            pass
        else:
            self.arrows = 0
            self.reward -= 40
            self.ij_step_flag = True
            tp_flag = True
        return tp_flag

    def gather(self):
        # if not np.array_equal(self.state, IJones.STATES.S):
            
        if np.array_equal(self.state, IJones.STATES.S) and probabilityTF(HIT_PROBABILITY):
            if self.materials == 0:
                self.materials += 1

    def attackMonster(self, monster, action: int):
        prob = 1

        if action == IJones.ATTACKS.HIT:
            if np.array_equal(self.state, StateOfPlayer.C):
                prob = BLADE_HIT_PROB_C
            elif np.array_equal(self.state, StateOfPlayer.E):
                prob = BLADE_HIT_PROB_E
        elif action == IJones.ATTACKS.SHOOT:
            self.arrows -= 1
            if self.arrows <= 0:
                return
            if np.array_equal(self.state, StateOfPlayer.E):
                prob = ARROW_HIT_PROB_E
            elif np.array_equal(self.state, StateOfPlayer.C):
                prob = ARROW_HIT_PROB_C
            elif np.array_equal(self.state, StateOfPlayer.W):
                prob = ARROW_HIT_PROB_W

        if probabilityTF(prob):
            print("Player attacked")
            monster.monsterAttacked(action)

    def craft(self):
        if np.array_equal(self.state, IJones.STATES.N) and self.materials != 0:
        #     pass
        # else:
            self.arrows += np.random.choice([1, 2, 3], p=[PROB_1_ARROW, PROB_2_ARROW, PROB_3_ARROW])
            self.materials -= 1

    
def main():
    os.makedirs("outputs", exist_ok=True)
    # print("Params", "DELTA", DELTA, "GAMMA", GAMMA, "STEP_COST", STEP_COST)
    # file = open("trace1.txt", "w")
    IJ = IJones()
    monster = MightyMonster()
    max_moves = 2000
    for iter_count in range(1,max_moves+1):
        # iter_count += 1
        # temp = "iteration="+ str(iter_count)+"\n"
        # file.write(temp)
        print("iteration="+ str(iter_count))
        monster.take_step(IJ)
        IJ.take_step(monster)
        # file.write(f"({IJ.current_state()},{IJ.materials},{IJ.arrows},{monster.state},{monster.health}):{IJ.action()}=[{IJ.values[IJ.current_state()]}]\n")
        print(f"({IJ.current_state()},{IJ.materials},{IJ.arrows},{monster.state},{monster.health}):{IJ.action()}=[{IJ.values[IJ.current_state()]}]")
        print("Monster health = " + str(monster.health))
        if monster.health <= 0:
            break

    return

player_board = np.array([
    [
    #W
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    [
    #N
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0.85, 0, 0, 0.15],
        [0, 0, 0.85, 0, 0.15],
        [0, 0, 0, 0, 0],
    ],
    [
    #C
        [0.85, 0, 0, 0, 0.15],
        [0, 0.85, 0, 0, 0.15],
        [0, 0, 0.85, 0, 0.15],
        [0, 0, 0, 0.85, 0.15],
        [0, 0, 0, 0, 1],
    ],
    [
    #S
        [0, 0, 0, 0, 0],
        [0, 0, 0.85, 0, 0.15],
        [0, 0, 0, 0.85, 0.15],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    [
    #E
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
])


main()
