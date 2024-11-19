#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# Building out the Model
from AgentDQN import AgentEnvironment, AgentMemory, AgentDQN
from DynamicProgramming import value_iteration
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import streamlit as st
import json
import pandas as pd

# Building out the model
class DQNBuild:
    def __init__(self, environment, cost, num_states, num_actions, hidden_dim=64, gamma = 0.9, lr=0.01, memory_length = 1000):
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

    def train(self, episodes, steps_per_episode, training_cost=1e3, batch_size=32, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995, episodic=False, terminal_period = False,
             use_target_network = False, target_update_period = None, convergence_stop = False):
        # initialize old policy to be used for convergence check
        previous_policy = np.zeros(shape=(1, self.num_states))
        state = self.env.reset() # incase episodic is False

        training_cost = training_cost
        total_cost = 0
        
        # initialize columns for graphing information
        #cols_ = st.columns([1,2,1,1])
        
        # Initialize columns for graphing information
        cols_ = st.columns(4)
        
        # Create subheaders and placeholders for each column
        with cols_[0]:
            st.subheader("Epsilon Decay")
            epsilon_placeholder = st.empty()  # Placeholder for epsilon chart
        
        with cols_[1]:
            st.subheader("Total Cost")
            total_cost_placeholder = st.empty()  # Placeholder for total cost chart
        
        with cols_[2]:
            st.subheader("Training Cost")
            training_cost_placeholder = st.empty()  # Placeholder for training cost chart
        
        with cols_[3]:
            st.subheader("Loss")
            loss_placeholder = st.empty()  # Placeholder for loss chart

        # create array to store epsilon, total_cost, training_cost, losses

        epsilons = []
        total_costs = []
        training_costs=[]
        
        for episode in range(episodes):
            terminal_period_counter = 0
            losses = []
            epochs = []
            
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

                    # append losses array
                    losses.append(loss)
                    epochs.append((episode+1)*step_)

                    #for i in range(len(losses)):
                    
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(epochs[:step_+1], losses[:step_+1])
                #ax.set_ylim(0,4000)   
                loss_placeholder.pyplot(fig)
                plt.close(fig)

               # losses_df = pd.DataFrame(epochs, index=losses, columns=["Epoch"])
                    # plot the losses
                #loss_placeholder.line_chart(losses, height=400)
                    


                # update state and total cost
                state = next_state
                training_cost -= cost
                total_cost += cost
                # epsilon decay
                epsilon = max(epsilon_min, epsilon*epsilon_decay)

                # append other parameters
                epsilons.append(epsilon)
                total_costs.append(total_cost)
                training_costs.append(training_cost)

                # plot the graphs
                epsilon_placeholder.line_chart(epsilons)
                total_cost_placeholder.line_chart(total_costs)
                training_cost_placeholder.line_chart(training_costs)

                

            # check for convergence
            # obtain current policy
            current_policy = self.get_policy()
            
            if convergence_stop:
                if np.array_equal(current_policy, previous_policy):
                    st.subheader(f'Convergence Reached With Statble Policy at {episode}')
                    st.subheader(f'Optimal Policy: {current_policy}')
                    break
                else:
                    previous_policy = current_policy
                
                
                
            # reset the starting state if it is the case the the system resets every year
            if episodic:
                state = self.env.reset()
                
        if convergence_stop != True:
            st.subheader(f'Current Policy {current_policy}')

    def get_policy(self):
        # check if the current policy is equal to the new policy
        id_vector = np.eye(self.num_states)
        q_values = self.q_model(id_vector)
        return np.argmin(q_values, axis=1)



# Sample data for testing purposes
st.set_page_config(layout='wide')

# Title and description
st.title('DQN Test Environment (Beta) for Single Agent')
st.write('This environment is set up to test the DQN performance against a baseline Value Iteration Algorithm as defined in Dynamic Programming.')

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
col1, col2 = st.columns(2)
with col1:
    num_actions = st.number_input('Number of actions', min_value = 0)

with col2:
    num_states = st.number_input('Number of states', min_value = 0)

def unpack_data(transition_matrix, cost, num_actions):
    # convert all
    for agent in range(1):
        transition_matrix[agent] = {int(k):v for k, v in transition_matrix[agent].items()}
        cost[agent] = {int(k) : v for k, v in cost[agent].items()}
    
        for action in range(num_actions):
            transition_matrix[agent][action] = pd.DataFrame(json.loads(transition_matrix[agent][action]))
            # obtain the columns
            agent_action_cols = [float(i) for i in transition_matrix[agent][action].columns.tolist()]
            transition_matrix[agent][action].columns = agent_action_cols
            cost[agent][action] = {int(k) : v for k, v in cost[agent][action].items()}

    return transition_matrix, cost

if uploaded_file is not None:
    agent_information = load_json(uploaded_file)
    transition_matrix, cost = agent_information
    transition_matrix, cost = unpack_data(transition_matrix=transition_matrix, 
                                         cost = cost, num_actions=num_actions)

    # generate columns dynamically based on the number of actions inputed
    columns = st.columns(num_actions)

    # obtain the transition information
    transition_matrix_ = transition_matrix[0]
    cost_ = cost[0]

    for action in range(num_actions):
        with columns[action]:
            st.subheader(f'Action {action}')
            st.write('Transition Matrix')
            st.dataframe(transition_matrix_[action])
            st.write('Cost')
            st.write(cost_[action])


    if st.button('Train'):
        baseline = value_iteration(environment=transition_matrix_, cost=cost_, n_actions=num_actions, tol=0.00001, n_state=num_states)
        st.subheader(f'Baseline Model (Value Iteration Algorithm): {baseline}')
        
        # use default values instead
        agent_dqn = DQNBuild(environment=transition_matrix_, cost = cost_, num_actions =num_actions, num_states=num_states)
        st.write('Agent Initialized')
        agent_dqn.train(episodes=5, steps_per_episode=365, batch_size=32, epsilon=1, use_target_network=True, target_update_period=10, convergence_stop=True)


    

