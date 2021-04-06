from collections import OrderedDict
from typing import Iterable
import os
import random
import numpy as np
# from dict import DotDict, npdict

# TEAM 44 - YogiJi
GAMMA = 0.999
DELTA = 1e-3
Y = [1/2, 1, 2][44 % 3]
STEP_COST = -10/Y
MOVEMENT_PROBABILITY = 0.85
HIT_PROBABILITY = 0.75
HEALING_POINTS = 25
MONSTER_STATE_CHANGE_PROB = 0.2
ATTACK_MONSTER_PROBABILITY = 0.5
MONSTER_DAMAGE = 25


def probabilityTF(prob: float):
    tp = random.random()
    if tp >= prob:
        return False
    else:
        return True

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
    "UP": tuple(PlayerTakeStep.UP),
    "DOWN": tuple(PlayerTakeStep.DOWN),
    "LEFT": tuple(PlayerTakeStep.LEFT),
    "RIGHT": tuple(PlayerTakeStep.RIGHT),
    "STAY": tuple(PlayerTakeStep.STAY),

})

class StateOfPlayer:
    W = np.array((-1, 0))
    N = np.array((0, 1))
    C = np.array((0, 0))
    S = np.array((0, -1))
    E = np.array((1, 0))

PlayerStates = DotDict({
    "N": tuple(StateOfPlayer.N),
    "S": tuple(StateOfPlayer.S),
    "E": tuple(StateOfPlayer.E),
    "W": tuple(StateOfPlayer.W),
    "C": tuple(StateOfPlayer.C),
})


_INV_STATE_MAP = {PlayerStates[k]: k for k in PlayerStates}
_INV_STATE_MAP_INDEX = {k: i for i, k in enumerate(PlayerStates)}


class IndexOfState:
    W = 0
    N = 1
    C = 2
    S = 3
    E = 4
    _inv_map = DotDict({
        tuple(StateOfPlayer.N): N,
        tuple(StateOfPlayer.S): S,
        tuple(StateOfPlayer.E): E,
        tuple(StateOfPlayer.W): W,
        tuple(StateOfPlayer.C): C,

    })

    def source_state(state):
        return IndexOfState._inv_map[tuple(state)]

_INV_ACTIONS_MAP = {PlayerActions[k]: k for k in PlayerActions}
_INV_ACTIONS_MAP.update({PlayerMoves[k]: k for k in PlayerMoves})
_INV_ACTIONS_MAP.update({PlayerAttacks[k]: k for k in PlayerAttacks})


class IJones(object):
    MOVES = PlayerMoves
    STATES = PlayerStates
    ATTACKS = PlayerAttacks
    ACTIONS = PlayerActions

    def __init__(self, inital_state=StateOfPlayer.N):
        self.state = inital_state
        self._action = IJones.ACTIONS.NONE
        self.arrows = 0
        self.materials = 0
        self.reward = 0
        self.stunned = False
        self._values = np.zeros((5,), dtype=np.float64)

    def move(self, direction):
        # add direction to state
        self.state += direction
        if np.sum(np.abs(self.state)) > 1:
            # out of bounds, illegal move -> undo move
            self.state -= direction

    # @property
    def current_state(self) -> str:
        return _INV_STATE_MAP[tuple(self.state)]

    # @property
    def action(self) -> str:
        return _INV_ACTIONS_MAP[self._action]

    @property
    def values(self):
        return npdict(self._values, _INV_STATE_MAP_INDEX)

    def move_to_E(self):
        self.state = StateOfPlayer.E

    def try_move(self, direction: tuple):
        if tuple(self.state) in [IJones.STATES.C, IJones.STATES.N, IJones.STATES.S]:
            if not probabilityTF(MOVEMENT_PROBABILITY):
                self.move_to_E()
                return
        # any other move for any state is determined i.e. prob = 1
        self.move(direction)

    def value_iteration(self):
        state_index = IndexOfState.source_state(self.state)
        rs = np.zeros((5,), dtype=np.float64)
        fxs = np.zeros((5,), dtype=np.float64)
        for i in range(5):
            rs[i] = np.sum(player_board[state_index][i] * STEP_COST)
            fxs[i] = GAMMA * np.sum(self._values.copy() * player_board[state_index][i])
        lst = rs + fxs
        maxcv = np.max(lst)
        self._values[state_index] = maxcv

    def get_wrecked(self):
        if tuple(self.state) not in [IJones.STATES.C, IJones.STATES.E]:
            return
        self.arrows = 0
        self.reward -= 40
        self.stunned = True

    def attackMonster(self, enemy, action: int):
        temp = 1
        if action == IJones.ATTACKS.SHOOT:
            if self.arrows == 0:
                return
            self.arrows -= 1
            if np.array_equal(self.state, StateOfPlayer.C):
                temp = 0.5
            elif np.array_equal(self.state, StateOfPlayer.W):
                temp = 0.25
            elif np.array_equal(self.state, StateOfPlayer.E):
                temp = 0.9
        elif action == IJones.ATTACKS.HIT:
            if np.array_equal(self.state, StateOfPlayer.C):
                temp = 0.1
            elif np.array_equal(self.state, StateOfPlayer.E):
                temp = 0.2

        # deal appropriate damage with success_probability
        if probabilityTF(temp):
            enemy.monsterAttacked(action)

    def craft(self):
        if not np.array_equal(self.state, IJones.STATES.N) or self.materials == 0:
            return
        self.materials -= 1
        self.arrows += np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

    def gather(self):
        if not np.array_equal(self.state, IJones.STATES.S):
            return
        if probabilityTF(HIT_PROBABILITY):
            if self.materials == 0:
                self.materials += 1

    def take_step(self, enemy):
        self.value_iteration()
        if self.stunned:
            # can't make any move for one round
            self.stunned = False
            return
        if not hasattr(self, 'choices'):
            # all possible choices player could make
            self.choices = []
            self.choices.extend(list(IJones.ATTACKS.values()))
            self.choices.extend(list(IJones.MOVES.values()))
            self.choices.extend(list(IJones.ACTIONS.values()))
        # perform a random action
        self._action = random.choice(self.choices)
        self.perform_action(enemy)

    def perform_action(self, enemy):
        action = self._action
        if action in IJones.ATTACKS.values():
            self.attackMonster(enemy, action)
        elif action in IJones.MOVES.values():
            self.try_move(random.choice(list(IJones.MOVES.values())))
        elif action in IJones.ACTIONS.values():
            if action == IJones.ACTIONS.CRAFT:
                self.craft()
            elif action == IJones.ACTIONS.GATHER:
                self.gather()
            elif action == IJones.ACTIONS.NONE:
                pass


class MightyMonster(object):
    STATES = DotDict({"D": 0, "R": 1})
    def __init__(self):
        self.health = 100
        self.state = MightyMonster.STATES.D

    def monsterAttacked(self, damage: int = MONSTER_DAMAGE):
        self.health -= damage

    def try_attackMonster(self, player: IJones):
        if probabilityTF(ATTACK_MONSTER_PROBABILITY):
            # attackMonster
            player.get_wrecked()
            self.health += HEALING_POINTS  # healing of monster
            self.state = MightyMonster.STATES.D  # change monster state to dormant

    def take_step(self, player: IJones):
        if probabilityTF(MONSTER_STATE_CHANGE_PROB): 
            self.state = MightyMonster.STATES.R  # change monster state to ready
        if self.state == MightyMonster.STATES.R:
            self.try_attackMonster(player)
def main():
    # file = open("trace1.txt", "w")
    IJ = IJones()
    monster = MightyMonster()
    max_moves = 1000
    for iter_count in range(1,max_moves+1):
        # iter_count += 1
        # temp = "iteration="+ str(iter_count)+"\n"
        # file.write(temp)
        print("iteration="+ str(iter_count))
        monster.take_step(IJ)
        IJ.take_step(monster)
        # file.write(f"({IJ.current_state()},{IJ.materials},{IJ.arrows},{monster.state},{monster.health}):{IJ.action()}=[{IJ.values[IJ.current_state()]}]\n")
        print(f"({IJ.current_state()},{IJ.materials},{IJ.arrows},{monster.state},{monster.health}):{IJ.action()}=[{IJ.values[IJ.current_state()]}]")
        if monster.health <= 0:
            break

    return

#       W    N     C    S    E
#   L
#   U
#   S
#   D
#   R
player_board = np.array([
    [
    #W
        [.00, .00, .00, .00, .00],
        [.00, .00, .00, .00, .00],
        [1.0, .00, .00, .00, .00],
        [.00, .00, .00, .00, .00],
        [.00, .00, 1.0, .00, .00],
    ],
    [
    #N
        [.00, .00, .00, .00, .00],
        [.00, .00, .00, .00, .00],
        [.00, .85, .00, .00, .15],
        [.00, .00, .85, .00, .15],
        [.00, .00, .00, .00, .00],
    ],
    [
    #C
        [.85, .00, .00, .00, .15],
        [.00, .85, .00, .00, .15],
        [.00, .00, .85, .00, .15],
        [.00, .00, .00, .85, .15],
        [.00, .00, .00, .00, 1.0],
    ],
    [
    #S
        [.00, .00, .00, .00, .00],
        [.00, .00, .85, .00, .15],
        [.00, .00, .00, .85, .15],
        [.00, .00, .00, .00, .00],
        [.00, .00, .00, .00, .00],
    ],
    [
    #E
        [.00, .00, 1.0, .00, .00],
        [.00, .00, .00, .00, .00],
        [.00, .00, .00, .00, 1.0],
        [.00, .00, .00, .00, .00],
        [.00, .00, .00, .00, .00],
    ],
])

# os.makedirs("outputs", exist_ok=True)

main()
