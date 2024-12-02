#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


def value_iteration(environment, cost, n_actions, tol, n_state, gamma):
    # Initialize the value function (V) and policy arrays
    v_new = np.zeros(shape=(n_state, 1))

    # initialize the starting policy
    policy_o = np.zeros(shape=v_new.shape)
    # initialize array to store new policy
    policy_n = policy_o.copy()
    
    while True:
        v_n = v_new.copy()
        
        for state in range(n_state):
            action_value = []
            for action in range(n_actions):
                # Obtain the cost of the action for the current state
                c_i = cost[action][state]
                # Obtain the transition probabilities for the current state and action
                p_ij = environment[action].loc[state].values
                # Calculate the value for the state-action pair
                v_ = (c_i + gamma* np.dot(p_ij.reshape(1, -1), v_n)).item()
                # Append the action value
                action_value.append(v_)

            # Update v_new for the current state with the minimum action value
            v_new[state] = np.min(action_value)
            # Update the policy with the action that yields the minimum value
            policy_n[state] = np.argmin(action_value)

           # print('v_new and v_check:{}'.format(v_new - v_n))

        # Calculate convergence bounds
        m_n = np.min(v_new - v_n)  # Lower bound
        M_n = np.max(v_new - v_n)  # Upper bound

        # Convergence check
        if (np.max(np.abs(v_new - v_n)) <= tol) or np.array_equal(policy_n, policy_o):
            break

        # Copy the current values to v_n for this iteration
        policy_o = policy_n
    return policy_n.flatten()

