# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import json
import pandas as pd
from collections import deque
import random
import streamlit as st

from AgentDQN import AgentEnvironment, AgentDQN
from DynamicProgramming import value_iteration

import matplotlib.pyplot as plt


class AgentMemory:
    def __init__(self, max_length):
        self.memory = deque(maxlen=max_length)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size, num_states):
        batch_info = random.sample(self.memory, batch_size)
        states, actions, next_states, costs, dones = zip(*batch_info)

        # onehot encode states and next states
        states = [int(s) for s in states]
        next_states = [int(ns) for ns in next_states]

        id_vector = np.eye(num_states)
        states = id_vector[states]
        next_states = id_vector[next_states]

        return states, np.array(actions).reshape(-1, 1), next_states, costs, np.array(dones) #np.array(dones).reshape(-1,1)

    def __len__(self):
        return len(self.memory)

class MultiAgentEnvironment:
    def __init__(self, num_agents, transition_matrices, costs, num_states):
        self.num_agents = num_agents
        # initialize the environment for each agent
        self.agents = [AgentEnvironment(environment=transition_matrices[agent], cost=costs[agent], num_states=num_states) for agent in range(self.num_agents)]

    def reset(self):
        # obtain list of random start states for each agent
        return [agent.reset() for agent in self.agents] # list of start states for each agent shape num_agents by states

    def step(self, states, actions):
        next_states = [] # list of start states for each agent shape: num_agents by next_states (num_agents, 1)
        costs = []   # list of start states for each agent shape: num_agents by costs (num_agents, 1)
        dones = []  # list of start states for each agent shape: num_agents by dones  (num_agents, 1)

        for i, (state, action) in enumerate(zip(states, actions)): # states, actions, is a list of states and actions of all agents
            next_state, cost, done = self.agents[i].step(state, action)
            # append next states, costs, dones
            next_states.append(next_state)
            costs.append(cost)
            dones.append(done)

        return next_states, costs, dones

class AgentTransformation(tf.keras.Model):
    def __init__(self, hidden_dim=64):
        super(AgentTransformation, self).__init__()
        self.h1 = tf.keras.layers.Dense(hidden_dim, activation='linear', kernel_initializer= tf.keras.initializers.GlorotNormal())
        self.h2 = tf.keras.layers.Dense(16, activation='linear', kernel_initializer= tf.keras.initializers.GlorotNormal())
        self.h3 = tf.keras.layers.Dense(8, activation='linear', kernel_initializer= tf.keras.initializers.GlorotNormal())
        self.weight_ = tf.keras.layers.Dense(1, activation='softplus')#, kernel_initializer='he_normal', name='v_weights')
        self.bias_ = tf.keras.layers.Dense(1, activation='softplus')#, name='v-bias')

    def call(self, global_state, agent_values):
        x = self.h1(global_state)
        x = self.h2(x)
        x = self.h3(x)
        weight = self.weight_(x)
        bias = self.bias_(x)
        transformation_value = (weight*agent_values) + bias

        return transformation_value

class MixingNetwork(tf.keras.Model):
    def __init__(self, num_states, num_actions, num_agents, hidden_dim=64):
        super(MixingNetwork, self).__init__()
        self.num_states = num_states
        self.num_agents = num_agents
        # initialize the DQN for each agent to be used to obtain the Q values
        self.agents_dqns = [AgentDQN(num_states = num_states, num_actions=num_actions, hidden_dim = hidden_dim) for agent in range(num_agents)]

        # intialize the transformation network to be used to calculate the transformed a's and v's for each agent
        self.agents_v_transforms = [AgentTransformation(hidden_dim = hidden_dim) for agent in range(num_agents)]
        self.agents_a_transforms = [AgentTransformation(hidden_dim = hidden_dim) for agent in range(num_agents)]

        # create feedforward network to obtain lambda for each agent
        self.lambda_weights = tf.keras.layers.Dense(num_agents, activation='softplus')#), kernel_initializer='he_normal', name='lambda_weights')

    def call(self, states, actions, global_states, batch_size, target_network=False):
        # at this point each states and actions will have shape num_agents, batch_size, 1

        # calculate the q values for each agent
        q_values = [self.agents_dqns[agent](states[agent]) for agent in range(self.num_agents)] # shape num_agents, batch_size, 2
        # there might be an issue here

        # stack the q_values on the batch_size dimension to make the shape batch_size, num_agents, q_values
        q_values = tf.stack(q_values, axis=1)

        if target_network:
          # obtain the q-values based on the e-greedy actions
          q_action_chosen = tf.reduce_min(q_values, axis=-1) # shape(batch_size, num_agents)

        else:
          #stack the actions as well
          actions_stacked = tf.stack(actions, axis=1)

          # # create indices that will be used to extract the q-values for each agent
          batch_indices = tf.reshape(tf.range(batch_size), (-1,1)) # batch_size by 1
          agent_indices = tf.range(self.num_agents) # shape (num_agents, [1])

          # create the batch array to be used for extraction
          batch_array_indices = tf.stack(tf.meshgrid(batch_indices[:, 0], agent_indices, indexing='ij'), axis=-1) # batch_size, num_agents, 2 [i,j values]

          # add the action to the batch_array indices in order to get the corresponding q_values
          actions_stacked = tf.cast(actions_stacked, dtype=tf.int32)
          batch_array_indices = tf.concat([batch_array_indices, actions_stacked], axis=-1)

          # obtain the q-values based on the e-greedy actions
          q_action_chosen = tf.gather_nd(q_values, batch_array_indices) # shape (batch_size, num_agents)

        # calculate the v which is taken as the minimum q value for each agent in each batch
        v_value = tf.reduce_min(q_values, axis=-1) # shape(batch_size, num_agents)

        # calculate the advantage
        agent_advantages = tf.expand_dims(q_action_chosen - v_value, axis=-1)
        v_value = tf.expand_dims(v_value, axis=-1)

        # Calculate the Transformed Advantage
        transformed_advantages = [
            self.agents_a_transforms[agent](global_state=global_states, agent_values=agent_advantages[:, agent]) for agent in range(self.num_agents)
        ] # num_agents, batch_size, 1
        # Calculate the transformed V - Value
        transformed_v_value = [
            self.agents_v_transforms[agent](global_state=global_states, agent_values=v_value[:, agent]) for agent in range(self.num_agents)
        ] # shape: num_agents, batch_size, 1

        # obtain the lambda weights
        transformed_adv = tf.stack(transformed_advantages, axis=1) # shape becomes batch_size, num_agents, 1
        transformed_values = tf.stack(transformed_v_value, axis=1) # shape = batch_size, num_agents, 1

        # calculate the lambda weight using the global state
        lambda_w = self.lambda_weights(global_states) # outputs a lambda for each agent
        lambda_w = tf.expand_dims(lambda_w, axis=-1)

        # calculate the joint q value
        joint_q_value = tf.reduce_sum(
            transformed_values + (lambda_w*transformed_adv), axis=1
        )

        return joint_q_value

class QPLEXBuild:
    def __init__(self, num_agents, num_states, num_actions, transition_matrices, costs, lr=0.001, memory_length=1000, hidden_dim=64, gamma=0.95):
        # initialize the parameters
        self.env = MultiAgentEnvironment(num_agents=num_agents, transition_matrices=transition_matrices, costs=costs, num_states=num_states)
        self.buffers = [AgentMemory(max_length = memory_length) for agent in range(num_agents)]
        self.qplex_model = MixingNetwork(num_states=num_states, num_actions=num_actions, num_agents=num_agents, hidden_dim=hidden_dim)
        self.target_qplex_model = MixingNetwork(num_states=num_states, num_actions=num_actions, num_agents=num_agents, hidden_dim=hidden_dim)

        # initialize training parameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_agents = num_agents
        self.gamma = gamma

        # train_step_counter for updating target network
        self.train_step_counter = 0

    def choose_action(self, agent_index, state, epsilon):
        # choose action based on epsilon-greedy probability
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            # onehot the received agent state
            state_onehot = tf.expand_dims(tf.one_hot(state, self.num_states), axis=0)
            # predict q_values #
            q_values = self.target_qplex_model.agents_dqns[agent_index](state_onehot)

            # select action with minimum q_value
            return tf.argmin(q_values, axis=1).numpy()[0]  # note that this is for a single agent

    def update_target_network(self):
        self.target_qplex_model.set_weights(self.qplex_model.get_weights())

    def train_step(self, states, actions, costs, next_states, global_states, global_next_states, batch_size, use_target_network, target_update_period, done = False):
        # initialize the loss
        mse_ = tf.keras.losses.MeanSquaredError()

        with tf.GradientTape() as tape:
            # obtaint the current q values
            current_joint_q = self.qplex_model(states=states, actions=actions, global_states=global_states, batch_size=batch_size) # batch_size, 1

            # calculate the target q values
            target_model = self.target_qplex_model if use_target_network else self.qplex_model
            #target_q_values = target_model(states=states, actions=actions, global_states=global_states, batch_size=batch_size).numpy() # this is y

            # calculate the backup q
            backup_joint_q = target_model(states=next_states, actions=actions, global_states=global_next_states,
                                          batch_size=batch_size, target_network=True).numpy() # batch_size, 1

            # calculate the target q with bellman equation
            target_joint_q = tf.convert_to_tensor(costs + self.gamma * backup_joint_q * (1-done))

            loss = mse_(target_joint_q, current_joint_q)

        # compute the gradient
        grads = tape.gradient(loss, self.qplex_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.qplex_model.trainable_variables))

        # update the target network counted
        self.train_step_counter += 1
        if use_target_network and target_update_period and self.train_step_counter % target_update_period == 0:
            self.update_target_network()

        return loss

    def train(self, episodes, steps_per_episode, training_cost=1e5, batch_size=32, epsilon=1, epsilon_min=0.1, epsilon_decay=0.9995, episodic=False, terminal_period = False,
             use_target_network = False, target_update_period = None):
        # initialize epsilon for all agents
        epsilon_per_agent = [epsilon for agent in range(self.num_agents)]
        # initialize old policy - do this later
        previous_policy_per_agent = np.random.rand(self.num_agents, self.num_states)

        training_cost = [training_cost for agent in range(self.num_agents)]
        total_cost = [0 for agent in range(self.num_agents)]


        # initialize state in case it is not episodic
        states = self.env.reset()


        # Initialize columns for graphing information
        cols_ = st.columns(4)

        # create array to store epsilon, total_cost, training_cost, losses
        total_costs = []
        training_costs = []
        
        # Create subheaders and placeholders for each column
        with cols_[0]:
            st.subheader("Epsilon Decay Per Agent")
            epsilon_placeholder = st.empty()  # Placeholder for epsilon chart
        
        with cols_[1]:
            st.subheader("Acc Cost Per Agent")
            total_cost_placeholder = st.empty()  # Placeholder for total cost chart
        
        with cols_[2]:
            st.subheader("Training Cost Red. Per Agent")
            training_cost_placeholder = st.empty()  # Placeholder for training cost chart
        
        
        with cols_[3]:
            st.subheader("Centralized Loss")
            loss_placeholder = st.empty()  # Placeholder for loss chart

        epsilons_per_ag = {0: []}
        total_costs_per_ag = {0: [], 1: []}
        training_costs_per_ag = {0: [], 1: []}

        # create the dfs
        epsilons_per_ag_df = pd.DataFrame(epsilons_per_ag)
        total_costs_per_ag_df = pd.DataFrame(total_costs_per_ag)
        training_costs_per_ag_df = pd.DataFrame(training_costs_per_ag)


        
        for episode in range(episodes):
            terminal_period_counter = 0
            losses = []
            epochs = []

            for step_ in range(steps_per_episode):
                terminal_period_counter += 1
                
                # each agent chooses an action
                actions = [self.choose_action(agent_index=agent, state=states[agent], epsilon=epsilon_per_agent[agent]) for agent in range(self.num_agents)] # shape: num_agents, 1

                # obtain the one step transition information
                next_states, costs, _ = self.env.step(states=states, actions=actions) # no need for the dones per agent. shape for each one (num_agents, 1)

                # store in memory
                for agent_ in range(self.num_agents):
                    self.buffers[agent_].add((states[agent_], actions[agent_], next_states[agent_], costs[agent_], _[agent_]))
                
                # train if enough samples
                if len(self.buffers[0]) >= batch_size:
                    # sample experiences
                    batch_info = [buffer.sample(batch_size, num_states=self.num_states) for buffer in self.buffers] # shape (num_agents, batch_size)

                    # obtain batch information
                    batch_states = np.array([data[0] for data in batch_info]) # num_agents, batch_size, [1]
                    batch_actions = np.array([data[1] for data in batch_info])
                    batch_next_states = np.array([data[2] for data in batch_info]) # num_agents, batch_size, [1]
                    batch_costs_ = [data[3] for data in batch_info]

                    # calculate the joint cost
                    joint_bacth_costs = tf.expand_dims(tf.reduce_sum(tf.stack(batch_costs_, axis=1), axis=1), axis=-1)

                    # create the global_current_state
                    batch_global_state = tf.stack(batch_states.tolist(), axis=1) # batch_size, num_agents
                    batch_global_next_state = tf.stack(batch_next_states.tolist(), axis=1) # batch_size, num_agents

                    # reshape the global and global next states so that its batch size, num_states*2
                    batch_global_state = tf.reshape(batch_global_state, (batch_global_state.shape[0], -1))
                    batch_global_next_state = tf.reshape(batch_global_next_state, (batch_global_next_state.shape[0], -1))

                    # update the dones if episodic
                    done = False
                    if terminal_period and terminal_period_counter % terminal_period == 0:
                        done = True
                    if terminal_period == False and episode == episodes-1 and step_ == steps_per_episode:
                        done = True

                    # calculate the loss
                    loss = self.train_step(states =batch_states, actions=batch_actions, next_states=batch_next_states, costs=joint_bacth_costs,
                                          global_states = batch_global_state, global_next_states=batch_global_next_state, batch_size=batch_size,
                                          use_target_network=use_target_network, target_update_period=target_update_period, done=done)
                    losses.append(loss)
                    epochs.append((episode+1)*step_)
                    #print(f'Running: {loss}')

                # plot the losses
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(epochs[:step_+1], losses[:step_+1])
                ##ax.set_ylim(0,20000)   
                loss_placeholder.pyplot(fig)
                plt.close(fig)

                # updates
                epsilons_per_ag_new = {}
                total_costs_per_ag_new = {}
                training_costs_per_ag_new = {}
                epsilons_per_ag_new[0] = max(epsilon_min, epsilon_per_agent[0]*epsilon_decay)
                for k in range(self.num_agents):
                    total_costs_per_ag_new[k] = total_cost[k] + costs[k]
                    training_costs_per_ag_new[k] = training_cost[k] - costs[k]
                

                #updated_dfs
                epsilons_per_ag_df = pd.concat([epsilons_per_ag_df, pd.DataFrame([epsilons_per_ag_new])], ignore_index=True)
                total_costs_per_ag_df = pd.concat([total_costs_per_ag_df, pd.DataFrame([total_costs_per_ag_new])], ignore_index=True)
                training_costs_per_ag_df = pd.concat([training_costs_per_ag_df, pd.DataFrame([training_costs_per_ag_new])], ignore_index=True)

                # updated_charts
                # plot the graphs
                epsilon_placeholder.line_chart(epsilons_per_ag_df)
                total_cost_placeholder.line_chart(total_costs_per_ag_df)
                training_cost_placeholder.line_chart(training_costs_per_ag_df)            
                

                # update states
                states = next_states
                # update training cost, total_cost, epsilon
                for ag_ in range(self.num_agents):
                    training_cost[ag_] -= costs[ag_]
                    total_cost[ag_] += costs[ag_]
                    epsilon_per_agent[ag_] = max(epsilon_min, epsilon_per_agent[ag_]*epsilon_decay)
                

            # check for convergence
            # obtain current policy
            current_policy = self.get_policy()
            #print('Current Policy for All Agents \n{}\n'.format(current_policy))
            if np.all(current_policy == previous_policy_per_agent):
                st.subheader(f'Convergence Reached With Statble Policy at {episode}')
             ##   print(f'Optimal Policy: {current_policy}')
                
                cols_opt = st.columns(num_agents)
                for ag__ in range(num_agents):
                    with cols_opt[ag__]:
                        st.subheader(f'Agent {ag__}:  {current_policy[ag__]}')
                break
            else:
                previous_policy_per_agent = current_policy

            # reset the starting state if it is the case the the system resets every year
            if episodic:
                state = self.env.reset()

        current_policy = self.get_policy()
        st.write(current_policy)

    def get_policy(self):
        # get the policy for each agent
        id_vector = np.eye(self.num_states)
        q_values_agents = []

        for agent in range(self.num_agents):
            q_vals = self.qplex_model.agents_dqns[agent](id_vector)
            q_vals = np.argmin(q_vals, axis=1)
            q_values_agents.append(q_vals)

        return np.array(q_values_agents)

def unpack_data(transition_matrix, cost, num_actions, num_agents):
    # convert all
    for agent in range(num_agents):
        transition_matrix[agent] = {int(k):v for k, v in transition_matrix[agent].items()}
        cost[agent] = {int(k) : v for k, v in cost[agent].items()}

        for action in range(num_actions):
            transition_matrix[agent][action] = pd.DataFrame(json.loads(transition_matrix[agent][action]))
            # obtain the columns
            agent_action_cols = [float(i) for i in transition_matrix[agent][action].columns.tolist()]
            transition_matrix[agent][action].columns = agent_action_cols
            cost[agent][action] = {int(k) : v for k, v in cost[agent][action].items()}

    return transition_matrix, cost



## Build Out the Streamlit dashboard
# Sample data for testing purposes
st.set_page_config(layout='wide')

# Title and description
#st.title('DQN Test Environment (Beta) for Multi Agent')
st.title('CTDE vs. Decentralized Value Iteration DP')

# Instructions
st.write('Please upload your data as a JSON file. Entries of the JSON file should have the following entries: `agent_id`, `environment`, and `cost`. Each entry is a list item.')
st.write("Sample data is available for use.")

# Upload JSON file
uploaded_file = st.file_uploader("Choose a JSON file", type="json")


def load_json(file):
    file_content = file.read().decode('utf-8')

    try:
        return json.loads(file_content)
        
    except json.JSONDecodeError:
        st.error('Invalid JSON file format. Please upload a valid JSON file')
        return None

# ask for number of actions
col1, col2, col3 = st.columns(3)
with col1:
    num_actions = st.number_input('Number of actions', min_value = 0)

with col2:
    num_states = st.number_input('Number of states', min_value = 0)

with col3:
    num_agents = st.number_input('Number of agents', min_value = 0)


if uploaded_file is not None:
    agent_information = load_json(uploaded_file)
    transition_matrices, costs = agent_information
    transition_matrices, costs = unpack_data(transition_matrix=transition_matrices, 
                                         cost = costs, num_actions=num_actions, num_agents=num_agents)
    
    transition_matrices = [transition_matrices[i] for i in range(num_agents)]
    costs = [costs[i] for i in range(num_agents)]


    if st.button('Train'):
        st.subheader(f'Baseline Model (Value Iteration)')
        cols_value = st.columns(num_agents)
        for agent in range(num_agents):
            with cols_value[agent]:
                baseline = value_iteration(environment=transition_matrices[agent], cost=costs[agent], n_actions=num_actions, tol=0.00001, n_state=num_states, gamma=0.9)
                st.subheader(f'Agent {agent}:  {baseline}')

        
        
        # use default values instead
        agents_qplex = QPLEXBuild(num_agents=num_agents, num_states=num_states, num_actions=num_actions, transition_matrices=transition_matrices, costs=costs)
        st.write('All Agents Initialized')
        ## changed epsilon here
        agents_qplex.train(episodes=10, steps_per_episode=365, use_target_network=True, target_update_period=10, epsilon_decay=0.995, batch_size=256,
                          epsilon=0, epsilon_min=0.0)




