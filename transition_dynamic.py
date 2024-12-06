import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(failure_path, maintenance_path):
    """
    Load and preprocess the failure and maintenance datasets.
    """
    # Load datasets
    failure_data = pd.read_csv(failure_path)
    maintenance_data = pd.read_csv(maintenance_path)
    
    # Convert timestamps
    failure_data['ActiveStartTime'] = pd.to_datetime(failure_data['ActiveStartTime'])
    failure_data['ActiveStopTime'] = pd.to_datetime(failure_data['ActiveStopTime'])
    maintenance_data['WO_Requested'] = pd.to_datetime(maintenance_data['WO_Requested'])
    
    return failure_data, maintenance_data

def preprocess_state_transitions(infusion_data, maintenance_data):
    """
    Create state transitions from infusion and maintenance data.
    """
    infusion_sorted = infusion_data.sort_values(['PCUSerialNumber', 'ActiveStartTime'])
    transitions = []
    
    for machine in infusion_sorted['PCUSerialNumber'].unique():
        machine_data = infusion_sorted[infusion_sorted['PCUSerialNumber'] == machine]
        
        for _, row in machine_data.iterrows():
            # Transition to active state
            transitions.append({
                'machine_id': machine,
                'current_state': 'INACTIVE',
                'next_state': row['WO_Type'],
                'timestamp': row['ActiveStartTime']
            })
            # Transition back to inactive
            transitions.append({
                'machine_id': machine,
                'current_state': row['WO_Type'],
                'next_state': 'INACTIVE',
                'timestamp': row['ActiveStopTime']
            })
    
    return pd.DataFrame(transitions)

def check_maintenance_transitions(transitions_df, maintenance_data):
    """
    Include maintenance events in transitions.
    """
    updated_transitions = []
    
    for _, transition in transitions_df.iterrows():
        machine = transition['machine_id']
        current_time = pd.to_datetime(transition['timestamp'])
        
        # Find next transition for this machine
        next_transitions = transitions_df[
            (transitions_df['machine_id'] == machine) & 
            (pd.to_datetime(transitions_df['timestamp']) > current_time)
        ]
        
        if len(next_transitions) > 0:
            next_time = pd.to_datetime(next_transitions.iloc[0]['timestamp'])
            
            # Check for maintenance during this period
            maintenance_events = maintenance_data[
                (maintenance_data['Asset_Serial'] == machine) &
                (pd.to_datetime(maintenance_data['WO_Requested']) > current_time) &
                (pd.to_datetime(maintenance_data['WO_Requested']) < next_time)
            ]
            
            if len(maintenance_events) > 0:
                for _, maint in maintenance_events.iterrows():
                    updated_transitions.append({
                        'machine_id': machine,
                        'current_state': transition['current_state'],
                        'next_state': 'MAINTENANCE',
                        'timestamp': maint['WO_Requested']
                    })
            
        updated_transitions.append(transition.to_dict())
    
    return pd.DataFrame(updated_transitions)

def calculate_transition_matrix(transitions_df):
    """
    Calculate the transition probability matrix.
    """
    all_states = sorted(list(set(
        transitions_df['current_state'].unique().tolist() +
        transitions_df['next_state'].unique().tolist()
    )))
    
    n = len(all_states)
    transition_counts = np.zeros((n, n))
    state_to_idx = {state: idx for idx, state in enumerate(all_states)}
    
    for _, transition in transitions_df.iterrows():
        current_idx = state_to_idx[transition['current_state']]
        next_idx = state_to_idx[transition['next_state']]
        transition_counts[current_idx][next_idx] += 1
    
    row_sums = transition_counts.sum(axis=1)
    transition_matrix = np.divide(transition_counts, row_sums[:, np.newaxis], 
                                where=row_sums[:, np.newaxis] != 0)
    
    return pd.DataFrame(transition_matrix, index=all_states, columns=all_states)

def plot_transition_matrix(transition_matrix, output_path='transition_matrix_heatmap.png'):
    """
    Create and save a heatmap visualization of the transition matrix.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(transition_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('State Transition Probability Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_transitions(transitions_df):
    """
    Analyze transition patterns and generate statistics.
    """
    stats = {
        'total_transitions': len(transitions_df),
        'unique_machines': transitions_df['machine_id'].nunique(),
        'state_counts': transitions_df['current_state'].value_counts().to_dict(),
        'maintenance_transitions': len(transitions_df[transitions_df['next_state'] == 'MAINTENANCE']),
        'avg_transitions_per_machine': len(transitions_df) / transitions_df['machine_id'].nunique()
    }
    return stats

def main():
    # File paths
    failure_path = 'sample_failure_info.csv'
    maintenance_path = 'sample_maint_data.csv'
    
    # Load and process data
    print("Loading data...")
    failure_data, maintenance_data = load_and_preprocess_data(failure_path, maintenance_path)
    
    print("Processing transitions...")
    transitions = preprocess_state_transitions(failure_data, maintenance_data)
    updated_transitions = check_maintenance_transitions(transitions, maintenance_data)
    
    print("Calculating transition matrix...")
    transition_matrix = calculate_transition_matrix(updated_transitions)
    
    print("Analyzing transitions...")
    stats = analyze_transitions(updated_transitions)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transitions.to_csv(f'transitions_{timestamp}.csv', index=False)
    transition_matrix.to_csv(f'transition_matrix_{timestamp}.csv')
    
    # Plot matrix
    plot_transition_matrix(transition_matrix, f'transition_matrix_heatmap_{timestamp}.png')
    
    # Print summary
    print("\nAnalysis Summary:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\nTransition Matrix:\n{transition_matrix}")
    print("\nResults have been saved to CSV files and heatmap visualization.")

if __name__ == "__main__":
    main()