#######Multi_Armed_Bandits######################################################################################
'''
Here we have 10 slots machines that we will run 10,000 times. 
Each slot machine has had successful payout according to the probs array
'''

import numpy as np
import csv, math, random
probs = [1703,1295,728,1196,2695,926,1112,2091,952,489]
rows = 10000
cols = 10
dataset = np.zeros((rows,cols))
for c in range(cols):
    for r in random.sample(list(range(rows)), probs[c]):
        dataset[r,c]=1

random_reward = 0
for r in range(rows):
    random_reward+= dataset[r, np.random.randint(cols)]

#print(random_reward) # our baseline: playing randomly the slots
#Result is 1325.0

###############################Generalizing this process################################################################################

#we keep track of which column is giving us the most rewards and we initialize with zeros.
#which column gives us the most rewards
#how many times we had tried each of those columns

times_tried = np.zeros(cols) 
reward = np.zeros(cols) 
for row in range(rows):
    row_values = list(np.zeros(cols))
    for col in range(cols):
        if times_tried[col]==0:
            row_values[col] = 1000
        else:
            row_values[col] = ((reward[col]/times_tried[col]) + (math.sqrt(2*math.log(row+1)/times_tried[col])))
    max_idx = row_values.index(np.max(row_values))
    times_tried[max_idx] += 1
    reward[max_idx] += dataset[row, max_idx]

print(random_reward)
print(sum(reward))