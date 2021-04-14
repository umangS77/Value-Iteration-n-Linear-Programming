# MDL - Assignment 2  
## Team 38 - YogiJi
- Umang Srivastava (2019101090)
- Kshitij Mishra (2019111014)

# Part 2
## Task 1
### Interpretation and Analysis: 

- Value of Step cost = -10
- According to file part_2_trace.txt, it took 114 iterations (0 to 113) for convergence. 
- The value of Gamma affects the gains made. A small value of Gamma leads to short-term gain policy and a big value of Gamma leads to large-term gain policy. But as gamma increases, value iteration will converge slower.
- We observe that Indiana Jones will try to avoid attacks from MM to avoid negative rewards. 
- IJ tries to avoid action STAY when in C. This may happen again due to a high negative reward of -40 when MM attacks IJ.
- IJ takes action SHOOT more when in position E or W.
- IJ always takes action HIT when in E.
- Number of SHOOTS are more than HITS as the probability of giving damage to MM using arrow is larger than blade.
- IJ can be termed Risk-Seeking since whenever it is in E, it tries to attack MM.
  
## Simulation
### 1. Start state is (W,0,0,D,100)
Sample output:
```
['W', 0, 0, 'D', 100] :  RIGHT
['C', 0, 0, 'D', 100] :  RIGHT
['E', 0, 0, 'D', 100] :  HIT
['E', 0, 0, 'D', 100] :  HIT
['E', 0, 0, 'D', 50] :  HIT
['E', 0, 0, 'D', 50] :  HIT
['E', 0, 0, 'R', 50] :  HIT
['E', 0, 0, 'D', 75] :  HIT
['E', 0, 0, 'D', 75] :  HIT
['E', 0, 0, 'D', 25] :  HIT
['E', 0, 0, 'D', 25] :  HIT
['E', 0, 0, 'D', 25] :  HIT
['E', 0, 0, 'D', 25] :  HIT
['E', 0, 0, 'R', 25] :  HIT
['E', 0, 0, 'D', 50] :  HIT
['E', 0, 0, 'D', 0] :  NONE
```
Analysis:   
After many simulations, we found that the terminal state is always ['E',0,0,'R',0]:NONE or ['E',0,0,'D',0]:NONE.
### 2. Start state is (C,2,0,R,100) 
Sample output:
```
['C', 2, 0, 'R', 100] :  UP
['N', 2, 0, 'R', 100] :  STAY
['N', 2, 0, 'R', 100] :  STAY
['N', 2, 0, 'R', 100] :  STAY
['N', 2, 0, 'D', 100] :  CRAFT
['N', 1, 3, 'D', 100] :  DOWN
['E', 1, 3, 'D', 100] :  SHOOT
['E', 1, 2, 'D', 75] :  SHOOT
['E', 1, 1, 'D', 50] :  SHOOT
['E', 1, 0, 'D', 25] :  HIT
['E', 1, 0, 'D', 25] :  HIT
['E', 1, 0, 'R', 25] :  HIT
['E', 1, 0, 'D', 50] :  HIT
['E', 1, 0, 'R', 50] :  HIT
['E', 1, 0, 'R', 50] :  HIT
['E', 1, 0, 'R', 50] :  HIT
['E', 1, 0, 'D', 75] :  HIT
['E', 1, 0, 'R', 25] :  HIT
['E', 1, 0, 'R', 25] :  HIT
['E', 1, 0, 'D', 50] :  HIT
['E', 1, 0, 'D', 0] :  NONE
```
Analysis:   
After many simulations, we found position E with all the combinations of all material values and R/D state  of MM in the terminal state 

## Task 2

### Case 1. Indiana now on the LEFT action at East Square will go to the West Square.  

We noticed an increase in number of iterations for convergence (from 114 to 118). Not much change was observed apart from a few actions were changed to LEFT.

### Case 2. The step cost of the STAY action is now zero.  

The number of iterations for convergence dropped to 107 ( 0 to 106 ). The number of  STAY action increased by a lot (from 2025 to 15268) as there is no penalty to stay. Also since the cost to STAY is 0, IJ tends to wait till MM becomes dormant to attack.

### Case 3. Change the value of gamma to 0.25  

The number of iterations for convergence drastically decreased from 114(0 to 113) to 8(0 to 7). This is because the Gamma value(=0.25) now is very lower than earlier value, thus making it easier to achieve the result in less iterations.

# Part 3 - Linear Programming
## Making matrix A
- We create a 5-tuple of all possible combinations of 5 positions of IJ, 3 values of materials, 4 numbers of arrows, 2 states of MM and 5 health levels of MM. In all, we have 5x3x4x2x5 = 600 states.
-  Then we get all possible states and state action pairs and allocate an unique id to each state.
- We initialize the matrix A as follow:
```
final_A=np.zeros((len(states),len(st_acn_p)))
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
```

## Finding the policy and analyzing the results
We initialize x, R and alpha arrays. Then we try to maximize the value of RX according to constraint AX = alpha, X>=0. To determine policy for each state, we pick the action with highest valid X value by iterating for every state.

## Can there be multiple policies? Why?
Yes, we can have multiple policies since, the (state, actions) with the same corresponding value in X are interchangeable.

## What changes can you make in your code to generate another policy?
- When Finding the highest value of X, we can either choose first highest value of X or last highest value of X. The alpha, A and R will remain unaffected.
- The order of actions also affects the policy when all actions have same reward. We can alter the sequence of action, thus changing policy. It will change A and R but not alpha.
- Changing factors like start state or Stepcost also affects the policy. This will not change A or alpha but will change R as it represents the expected reward for each action-state.
