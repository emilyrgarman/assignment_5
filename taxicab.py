#%%
# Use the pickled taxicab data

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('taxicab.pkl', 'rb') as file:
    taxicab = pickle.load(file)

# taxicab = pd.read_pickle('taxicab.pkl')
# - For the taxicab trajectory data, determine your state space and clean your sequences of cab rides.
states = set()
for trajectory in taxicab:
    states = states.union(set(trajectory))
states = list(states)
print('States:\n', np.array(states))

#%%
# - Compute the transition matrix for the taxicab data between neighborhoods in Manhattan. Plot it in a heat map. What are the most common routes?
S = len(states)
T = len(taxicab)
tr_counts = np.zeros((S, S))

# Compute transition counts
for trajectory in taxicab:
    seq = np.array(trajectory)
    for t in range(1, len(seq)):
        # Current and next tokens:
        x_tm1 = seq[t - 1] # previous state
        x_t = seq[t] # current state
        # Determine transition indices:
        index_from = states.index(x_tm1)
        index_to = states.index(x_t)
        # Update transition counts:
        tr_counts[index_to, index_from] += 1

print('Transition Counts:\n', tr_counts)

# Sum transition counts across row:
sums = tr_counts.sum(axis = 0, keepdims = True)
print('State proportions: \n', sums)

# Normalize the transition count matrix to get proportions:
tr_pr = np.divide(tr_counts, sums, out = np.zeros_like(tr_counts), where = sums != 0)
print('Transition Proportions: \n', tr_pr)

# Transition matrix
tr_df = pd.DataFrame(np.round(tr_pr, 2), index = states, columns = states)
print(tr_df)

# %%
plt.figure(figsize = (12, 10))
sns.heatmap(tr_pr,
            cmap = 'crest',
            square = True,
            xticklabels = states,
            yticklabels = states,
            cbar_kws = {'label': 'Transition Probability'})

plt.title('Transition Probabilities')
plt.xlabel('From State...')
plt.ylabel('...To State')
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
plt.show()

#%%
# - Explain why taxicabs are most likely order 1, and not 2 or more.

#%%
# - Starting at Hell's Kitchen, create a sequence of forecasts of where the cab is likely to be in 2, 3, 5, and 10 trips

np.random.seed(100)

initial_state = "Hell's Kitchen"
state_index = states.index(initial_state)

n_sim = 2

simulation = [initial_state]
for t in range(n_sim - 1):
    pr_t = tr_pr[:, state_index] # Transition probabilities at this state
    state_index = np.random.choice(len(states), p = pr_t) # Choose new state index
    simulation.append(states[state_index]) # Append new state to simulation

print(simulation)

# %%
# - Starting at Hell's Kitchen, create a sequence of forecasts of where the cab is likely to be in 2, 3, 5, and 10 trips
# Altering code to run n trips in a for loop

np.random.seed(100)

initial_state = "Hell's Kitchen"
state_index = states.index(initial_state)

n_sim = [2, 3, 5, 10]

for n in n_sim:
    state_index = states.index(initial_state)
    simulation = [initial_state]
    
    for t in range(n - 1):
        pr_t = tr_pr[:, state_index] # Transition probabilities at this state
        state_index = np.random.choice(len(states), p = pr_t) # Choose new state index
        simulation.append(states[state_index]) # Append new state to simulation
    
    print(f"{n} trips: {simulation}")

#%%
# - Starting at any neighborhood, iterate your forecast until it is no longer changing very much. Where do cabs spend most of their time working in Manhattan?

np.random.seed(5030)

initial_state = np.random.choice(states)
state_index = states.index(initial_state)

# Initial density
density = np.zeros(len(states))
density[state_index] = 1

sns.barplot(x = states, y = density).set(title = f'Forecast: 0')
plt.xticks(rotation = 90)
plt.show()

n_sim = 10

forecast = [initial_state]
for t in range(n_sim):
    density = tr_pr @ density
    forecast.append(density)
    sns.barplot(x = states, y = density).set(title = f'Forecast, period: {str(t+1)}')
    plt.xticks(rotation = 90)
    plt.show()

# After forecast period 5, it doesn't change very much.
# %%
