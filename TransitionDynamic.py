#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import numpy as np


# In[ ]:


class TransitionDynamic():
  def __init__(self, infusion_log, maintenance_log, col_name, states=['infusing', 'CECM', 'HZARC', 'CEIN', 'ADMIN','CEDS',  'COMP-R', 'RND']):
    self.infusion_log = infusion_log
    self.maintenance_log = maintenance_log
    self.col_name = col_name
    #self.preprocessed_infusion = self.data_preprocessing()
    self.states = states
    self.coded_state = {val:i for i, val in enumerate(self.states)}


  def infusion_preprocessing(self):
    # initializa the data
    data = self.infusion_log.copy()

    # state before maintenance
    data['CurrentState'] = 'infusing'

    # reorder the columns
    data = data[[self.col_name, 'CurrentState','ActiveStartTime', 'ActiveStopTime', 'TotalInfusionTime', 'TotalEqActiveTime',
                 'WO_Requested', 'WO_WO#', 'WO_Type']].sort_values(by=[self.col_name, 'ActiveStartTime'])

    # obtain the state of the system at time t+2 for every row
    data = data.assign(NextActiveTime = data.groupby(self.col_name)['ActiveStartTime'].shift(-1),
                       NextCurrentState = data.groupby(self.col_name)['CurrentState'].shift(-1))

    # update the values of the final state in the last equipment observation in each partition
    last_indices_NextActTime = data.groupby(self.col_name).tail(1).index
    data.loc[last_indices_NextActTime, 'NextActiveTime'] = data.loc[last_indices_NextActTime, 'WO_Requested']
    data.loc[last_indices_NextActTime, 'NextCurrentState'] = data.loc[last_indices_NextActTime, 'WO_Type']

    return data

  def maintenance_preprocessing(self, group_assets):
    '''The method preprocesses the data set by updating the WO_Closed information to allow for
      estimated duration of repair. It is assumed in the function that there can't overlapping transition between states
      e.g. An ongoing CEIN must be closed if the same equipment is being transferred for CECM in the ongoing repair cycle.
      If repairs have transitioned to another active state, then
    '''

    # obtain the preprocessed infusion data
    infusion_data = self.infusion_preprocessing().copy()

    # obtain the maintenance data
    maintenance_data = self.maintenance_log.copy()

    # remove any duplicates from the work order
    maintenance_data = maintenance_data.drop_duplicates(subset=['Asset_Serial', 'WO_WO#', 'WO_Requested', 'WO_Type', 'WO_Closed'])

    # obtain the future work order
    maintenance_data = maintenance_data.assign(Next_WO_Requested = maintenance_data.groupby('Asset_Serial')['WO_Requested'].shift(-1))

    # merge the maintenance data with the infusion_data
    infusion_data = infusion_data.groupby('WO_WO#').agg({'NextActiveTime' : 'unique'}).reset_index()

    infusion_data['NextActiveTime'] = [val[0] for val in infusion_data.NextActiveTime]

    # change next active time to date
    infusion_data['NextActiveTime'] = pd.to_datetime(infusion_data['NextActiveTime'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

    # merge maintenance data with infusion_data
    maintenance_data = pd.merge(maintenance_data, infusion_data, how='left', on='WO_WO#')

    # Assume no overlap in times between current state and next state
    maintenance_data['WO_Closed'] = maintenance_data[['WO_Closed', 'Next_WO_Requested', 'NextActiveTime']].min(axis = 1)

    # additional check for WO_Requested
    #maintenance_data['WO_Requested'] = maintenance_data[['WO_Requested', 'WO_Closed']].min(axis=1)

    # calculate the repair duration
    maintenance_data['dum_col'] = maintenance_data['WO_Closed']

    # update instances where the equipment was completed in the same day so repair duration is not equal to 0
    maintenance_data.loc[maintenance_data.WO_Requested == maintenance_data.WO_Closed, 'dum_col'] = maintenance_data.dum_col + pd.Timedelta(days=1)

    # calculate repair duration
    maintenance_data['RepairDuration'] = (maintenance_data['dum_col'] - maintenance_data['WO_Requested']).dt.total_seconds()

    # drop dum_cell
    maintenance_data = maintenance_data.drop('dum_col', axis=1)

    if group_assets:
      # aggregate the data
      maintenance_data = maintenance_data.groupby(['Asset_Serial', 'WO_WO#', 'WO_Requested', 'WO_Type', 'WO_Closed']).agg({'RepairDuration': 'sum'}).reset_index()

    # Additional checks for debugging
    maintenance_data['Flag'] = 'Good'
    maintenance_data.loc[maintenance_data.RepairDuration < 0, 'Flag'] = 'Red - Confirm WO_Requested Date'


    return maintenance_data


  def agent_environment(self, equipment_serial, df, maint_df, cols, duration_var = False, ctmc = False):
    '''
        equipment_serial: Serial number of the cols specified
        cols: name of the equipment type as listed in the infusion log (e.g PCUSerialNumber or ModuleSerialNumber)
        df: preprocessed infusion dataframe
        maint_df: work order dataframe
    '''
    if duration_var:
      df = df.assign(FutureTime = df.groupby(cols)[duration_var].shift(-1)) #double check this


    state_code = self.coded_state

    # define the states for each dataframe
    infusion_state = df.CurrentState.unique().tolist()
    workOrder_state = maint_df.WO_Type.unique().tolist()

    # all states
    full_state = self.states
    state_encoding = {val: i for i, val in enumerate(full_state)}

    # filter the dataframe
    filtered_df = df[df[cols] == equipment_serial]

    # transition lists - current, next, and future state
    from_state = []
    to_state = []
    future_state = []

    # transition time
    from_to_time = []
    to_future_time = []

    # starting parameters
    indx = filtered_df.index

    for idx in indx:
      # obtain the window between the next state and the future state
      gap_start = filtered_df.loc[idx, 'WO_Requested']
      gap_end = filtered_df.loc[idx, 'NextActiveTime']

      gap_start_state = filtered_df.loc[idx, 'WO_Type']
      gap_end_state = filtered_df.loc[idx, 'NextCurrentState']


      if gap_start != gap_end and gap_start_state != gap_end_state:
        current_state = filtered_df.loc[idx, 'CurrentState']
        next_state = filtered_df.loc[idx, 'WO_Type']

        from_state.append(current_state)
        to_state.append(next_state)

        # starting parameter for from_to_time
        if duration_var:
          from_to_time_val = filtered_df.loc[idx, duration_var]
          from_to_time.append(from_to_time_val)


        # retreive the respective work order
        maint_info = maint_df[
            (maint_df.Asset_Serial == equipment_serial) & (maint_df.WO_Requested >= gap_start) & (maint_df.WO_Closed <= gap_end)
        ].sort_values(by='WO_Requested')

        if len(maint_info) == 1:
          f_state = filtered_df.loc[idx, 'NextCurrentState']
          # update the future state
          future_state.append(f_state)

          # update the from_state and the to_state
          from_state.append(to_state[-1])
          to_state.append(future_state[-1])

          if duration_var:
            future_time_val = maint_info.RepairDuration.values.tolist()[0]
            to_future_time.append(future_time_val)

            # update from_to_time
            from_to_time.append(to_future_time[-1])


        else:
          if len(maint_info) == 0:
            break

          else:
            # obtain all future state (this is inclusive of present)
            states_info = maint_info.WO_Type.values.tolist()[1:]

            # append with the gap end state so the equipment can transition back to infusing after leaving maintenance
            states_info.append(gap_end_state)

            if duration_var:
              # obtain the timestamp
              durations = maint_info.RepairDuration.values.tolist()

            for state in range(len(states_info)):
              # append future_state
              future_state.append(states_info[state])

              # append from state and to state
              from_state.append(to_state[-1])
              to_state.append(future_state[-1])

              if duration_var:
                to_future_time.append(durations[state])

                # update from_to time
                from_to_time.append(to_future_time[-1])

      else:
         from_state.append(filtered_df.loc[idx, 'CurrentState'])
         to_state.append(filtered_df.loc[idx, 'WO_Type'])
         #future_state.append(filtered_df.loc[idx, 'NextCurrentState'])

         if duration_var:
          from_to_time.append(filtered_df.loc[idx, duration_var])

    # encode the from and to state
    from_state_enc = [self.coded_state[f] for f in from_state]
    to_state_enc = [self.coded_state[t] for t in to_state]

    n_state = len(self.states)

    # compute the matrix
    transition_time = pd.DataFrame([from_state_enc,to_state_enc, from_to_time], index=['i_state', 'j_state', 'transition_time']).transpose()

    # compute the dtmc
    transition_count = transition_time.groupby(['i_state', 'j_state']).size().unstack(fill_value = 0)

    #compute the probability dtmc
    transition_matrix_pd = transition_count.div(transition_count.sum(axis=1),axis=0)

    # initialize the dtmc numpy array
    transition_matrix_np = np.zeros(shape = (n_state, n_state))

    if ctmc:
      sum_time = transition_time.groupby(['i_state', 'j_state'])['transition_time'].sum().unstack(fill_value=0)

      # build the numpy transition matrix
      transition_time_matrix_np = np.zeros(shape=(n_state, n_state))

    state_i = transition_matrix_pd.index
    state_j = transition_matrix_pd.columns

    for i in state_i:
      for j in state_j:
        #print([i,j])
        transition_matrix_np[int(i),int(j)] = transition_matrix_pd.loc[i,j]

        if ctmc:
          transition_time_matrix_np[int(i),int(j)] = sum_time.loc[i,j]


    # store solutions in a dict
    solution_dict = {'dtmc_matrix': transition_matrix_np,
                     'dtmc_matrix_pandas': transition_matrix_pd,
                     'dtmc_count' : transition_count,
                     'from_state': from_state,
                     'to_state': to_state
                     }
    if ctmc:
      solution_dict['transition_time'] = transition_time_matrix_np
      solution_dict['transition_time_raw'] = transition_time

      equip_maint_info = maint_df[maint_df.Asset_Serial == equipment_serial]

      solution_dict['maint_info'] = equip_maint_info

    return solution_dict

  def system_environment(self, group_assets = False, duration_var = None, ctmc = False):
    df = self.infusion_preprocessing()
    maint_df = self.maintenance_preprocessing(group_assets)
    cols = self.col_name
    asset_serials = df[cols].unique().tolist() # obtain the list of equipment ids
    equipment_solution = {}

    for equipment_serial in tqdm(asset_serials, desc='Transtion Matrix Computation'):
      matrix_solution = self.agent_environment(equipment_serial = equipment_serial,
                                                     df = df, maint_df = maint_df, cols = cols,
                                                       duration_var = duration_var, ctmc = ctmc)

      equipment_solution[equipment_serial] = matrix_solution

    return equipment_solution

