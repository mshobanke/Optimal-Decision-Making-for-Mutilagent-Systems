#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from collections import deque
import random
import pandas as pd


# In[49]:


class AgentEnvironment:
    def __init__(self, environment, cost, num_states):
        self.environment = environment
        self.cost = cost
        self.num_states = num_states
        self.state = np.random.randint(0, self.num_states)

    def reset(self):
        # random action to cater for instances where some states were not visited by the agent
        n_actions = len(self.environment.keys())
        r_action = np.random.randint(0, n_actions)
        
        # obtain set of possible start states
        trans_mat = self.environment[r_action].index.tolist()

        # obtain the starting state
        self.state = np.random.choice(trans_mat)
        #self.state = np.random.randint(0, self.num_states)
        return self.state

    def step(self, state, action):
        # obtain the possible next states
        ## 
        #print('action {}'.format(action))
        #print('state {}'.format(state))
        
        pos_next_states = self.environment[action].columns.tolist()
        p_j = self.environment[action].loc[state].values
        next_state = np.random.choice(pos_next_states, p=p_j)
        cost = self.cost[action][state]
        done = False
        return next_state, cost, done


# In[50]:


class AgentMemory:
    def __init__(self, max_length):
        self.memory = deque(maxlen=max_length)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size, num_states):
        batch_info = random.sample(self.memory, batch_size)
        states, actions, next_states, costs, dones = zip(*batch_info)
        #print(' memory next_states {}'.format(next_states)

        # onehot encode states and next states
        states = [int(s) for s in states]
        next_states = [int(ns) for ns in next_states]

        id_vector = np.eye(num_states)
        states = id_vector[states]
        next_states = id_vector[next_states]
        
        return states, actions, next_states, costs, np.array(dones)

    def __len__(self):
        return len(self.memory)


# In[51]:


class AgentDQN(tf.keras.Model):
    def __init__(self, num_states, num_actions, hidden_dim):
        super(AgentDQN, self).__init__()
        # Define network layers
        self.h1 = tf.keras.layers.Dense(hidden_dim, activation='linear', kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.h2 = tf.keras.layers.Dense(16, activation='linear', kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.h3 = tf.keras.layers.Dense(8, activation = 'linear', kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.q_vals = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, state_onehot):
        # forward pass
        x = self.h1(state_onehot)
        x = self.h2(x)
        x = self.h3(x)
        q_values = self.q_vals(x)
        return q_values


# In[52]:


class DQNBuild:
    def __init__(self, environment, cost, num_states, num_actions, hidden_dim, gamma = 0.95, lr=0.01, memory_length = 1000):
        # initialize the environment
        self.env = AgentEnvironment(environment=environment, cost=cost,num_states=num_states)
        # initialize the AgentDQN
        self.q_model = AgentDQN(num_states = num_states, num_actions = num_actions, hidden_dim=hidden_dim)
        # initalize the target network
        self.target_q_model = AgentDQN(num_states = num_states, num_actions = num_actions, hidden_dim=hidden_dim)
        # intialize the agent memory
        self.buffer = AgentMemory(max_length = memory_length)
        # discount rate
        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.train_step_counter = 0
        #self.target_update_period = target_update_period

    def update_target_network(self):
        # update target network at intervals to match the weights of the q-network
        self.target_q_model.set_weights(self.q_model.get_weights())

    def choose_action(self, state, epsilon):
        # choose action based on epsolon-greedy strategy with (np.argmin) for cost minimization
        if np.random.rand() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            # onehot encode the state
            state_onehot = tf.expand_dims(tf.one_hot(state, self.num_states), axis=0)
            # predict q_values
            q_values = self.target_q_model(state_onehot)

            # select action with minimum q_value
            return tf.argmin(q_values, axis=1).numpy()[0] 


    def train_step(self, states, actions, costs, next_states, dones, batch_size, use_target_network, target_update_period):
        '''
            predicts the q_values of the current state and the y_values for the current state
            Args:
                states: array (one hot encoding of current state)
                next_states: array (one hot encoding of next state) 
                dones:
                costs: cost of each action in current state
                action: array of selected actions
        '''
        # obtain the loss function
        mse_ = tf.keras.losses.MeanSquaredError()

        with tf.GradientTape() as tape:

            # Calculate current Q-values
            current_q_values = self.q_model(states)
            #print(f'current q shape{current_q_values.shape}')
            
            # Get target Q-values base
            target_q_values = self.target_q_model(states).numpy() if use_target_network else  self.q_model(states).numpy()
            backup_q = self.target_q_model(next_states).numpy() if use_target_network else  self.q_model(next_states).numpy() # copy current q values
            
            # Calculate backup Q-values for next states
            #backup_q = self.q_model(next_states).numpy()
            #print(f'backup_q shape {backup_q.shape}')

            # Calculate target Q-values with Bellman equation
            target_q = costs + self.gamma * np.min(backup_q, axis=1) * (1-dones)
            target_q_values[np.arange(batch_size), actions]= target_q
            target_q_values = tf.convert_to_tensor(target_q_values)
            
            # Calculate loss
            loss = mse_(target_q_values, current_q_values)

        # Compute gradients
        grads = tape.gradient(loss, self.q_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))

        # update the target network counted
        self.train_step_counter += 1
        if use_target_network and target_update_period and self.train_step_counter % target_update_period == 0:
            self.update_target_network()

        return loss

    def train(self, episodes, steps_per_episode, training_cost=1e5, batch_size=32, epsilon=1, epsilon_min=0.01, epsilon_decay=0.9995, episodic=False, terminal_period = False,
             use_target_network = False, target_update_period = None):
        # initialize old policy to be used for convergence check
        previous_policy = np.zeros(shape=(1, self.num_states))
        state = self.env.reset() # incase episodic is False

        training_cost = training_cost
        total_cost = 0
        
        
        for episode in range(episodes):
            terminal_period_counter = 0
            
            for step_ in range(steps_per_episode):
                terminal_period_counter += 1
                # choose action
                action = self.choose_action(state=state, epsilon=epsilon)

                #obtain the one step transition information
                next_state, cost, done = self.env.step(state, action)

                # update the dones if episodic
                if terminal_period and terminal_period_counter % terminal_period == 0:
                    done = True
                if terminal_period == False and episode == episodes-1 and step_ == steps_per_episode:
                    done = True

                # store experience in memory
                self.buffer.add((state, action, next_state, cost, done))

                if len(self.buffer)>= batch_size:
                    # unpack the batch_info
                    states, actions, next_states, costs, dones = self.buffer.sample(batch_size, num_states=self.num_states)

                    # train the model
                    loss = self.train_step(states = states, actions = actions, 
                                           next_states = next_states, costs = costs, 
                                           dones = dones, batch_size=batch_size, 
                                           use_target_network = use_target_network,
                                          target_update_period = target_update_period)


                # update state and total cost
                state = next_state
                training_cost -= cost
                total_cost += cost
                # epsilon decay
                epsilon = max(epsilon_min, epsilon*epsilon_decay)

            # check for convergence
            # obtain current policy
            current_policy = self.get_policy()
            if np.array_equal(current_policy, previous_policy):
                print(f'Convergence Reached With Statble Policy at {episode}')
                print(f'Optimal Policy: {current_policy}')
                break
            else:
                previous_policy = current_policy
                
            # reset the starting state if it is the case the the system resets every year
            if episodic:
                state = self.env.reset()
            

    def get_policy(self):
        # check if the current policy is equal to the new policy
        id_vector = np.eye(self.num_states)
        q_values = self.q_model(id_vector)
        return np.argmin(q_values, axis=1)