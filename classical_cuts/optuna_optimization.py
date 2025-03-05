import optuna
import numpy as np
import csv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history

#reading the file
file=open("data_preprocessing/osiris_toydata_7.csv","r")

def read_file(file):
    
    #lists to collect the data from the columns in the file
    time=[] #time in seconds
    events=[] #no of pe events
    x_coor=[] #x coordinates
    y_coor=[] #y coordinates
    z_coor=[] #z coordinates
    MC=[] 


    #collecting the data from the file onto the lists
    def read_allevents(file):
        with file as df:
            data = csv.reader(df, delimiter=' ') #reading the file
            for rows in data:
                time.append(rows[0]) #collecting the 1st column i.e. time column into a list
                events.append(rows[1]) #collecting pe events column
                x_coor.append(rows[2]) #collecting the x coordinates column
                y_coor.append(rows[3]) #collecting the y coordinates column
                z_coor.append(rows[4]) #collecting the z coordinates column
                MC.append(rows[-1]) 

    read_allevents(file) 
        
    time_f =[] #time in seconds as float
    events_f = [] #no of pe events as float
    x=[] #x cooradinates as float
    y=[] #y cooradinates as float
    z=[] #z cooradinates as float
    MC_f=[]
    #function to convert the data to float
    def floatconv(list1,list2):
        for item in list1:
            # Remove formatting characters before converting to float
            cleaned_item = item.replace('%', '').replace('f', '')
            list2.append(float(cleaned_item))
    

    floatconv(time,time_f) #time list into float
    floatconv(events,events_f) # pe events into float 
    floatconv(x_coor,x) #x coordinates into float
    floatconv(y_coor,y) #y coordinates into float
    floatconv(z_coor,z) #z coordinates into float
    floatconv(MC,MC_f)#MC into float    
        
    energy=[] #no of pe events as energy in MeV

    #function to convert number of photoevents into energy by dividing by 280
    def enrg (events_f, list):
        
        div=280 #dividant required to convert pe events into energy
        
        for ele in events_f: 
            val=ele/div #dividing photoevents by 280 to get the energy
            list.append(val) #collecting all the converted energy into a list

    enrg(events_f, energy)

    # Define the wanted values
    wanted_values_1 = [0.0, 1.0, 2.0, 3.0]
    wanted_values_2 = [0.0, 3.0, 4.0, 5.0]

    # Create lists to store filtered data
    time_214 = []
    energy_214 = []
    x_214 = []
    y_214 = []
    z_214 = []
    MC_214 = []

    # Iterate over the data and filter out unwanted values
    for time_val, energy_val, x_val, y_val, z_val, MC_val in zip(time_f, energy, x, y, z, MC_f):
        if MC_val in wanted_values_1:
            time_214.append(time_val)
            energy_214.append(energy_val)
            x_214.append(x_val)
            y_214.append(y_val)
            z_214.append(z_val)
            MC_214.append(MC_val)

    # Create lists to store filtered data
    time_212 = []
    energy_212 = []
    x_212 = []
    y_212 = []
    z_212 = []
    MC_212 = []

    # Iterate over the data and filter out unwanted values
    for time_val, energy_val, x_val, y_val, z_val, MC_val in zip(time_f, energy, x, y, z, MC_f):
        if MC_val in wanted_values_2:
            time_212.append(time_val)
            energy_212.append(energy_val)
            x_212.append(x_val)
            y_212.append(y_val)
            z_212.append(z_val)
            MC_212.append(MC_val)

    return time_214, energy_214, x_214, y_214, z_214, MC_214, time_212, energy_212, x_212, y_212, z_212, MC_212

time_214, energy_214, x_214, y_214, z_214, MC_214, time_212, energy_212, x_212, y_212, z_212, MC_212 = read_file(file)

def objective_1(trial):
    # Define the hyperparameters to tune
    r_fv_214 = trial.suggest_float('r_fv_214', 800, 1500)
    t1_t2_214 = trial.suggest_float('t1_t2_214', 0.0000002, 0.00711)
    E_bi_up_214 = trial.suggest_float('E_bi_up_214', 1, 5)
    E_bi_low_214 = trial.suggest_float('E_bi_low_214', 0.1, 3)
    E_po_up_214 = trial.suggest_float('E_po_up_214', 0.5, 3)
    E_po_low_214 = trial.suggest_float('E_po_low_214', 0.1, 2)
    dist_214 = trial.suggest_float('dist_214', 250, 700)
    h_fv_214 = trial.suggest_float('h_fv_214', 1000, 1500)

    # Use NumPy arrays for vectorized operations
    time_np = np.array(time_214)
    energy_np = np.array(energy_214)
    x_np = np.array(x_214)
    y_np = np.array(y_214)
    z_np = np.array(z_214)
    MC_np = np.array(MC_214)

    # Define chunk size for processing data
    chunk_size = 15000

    signal_count = 0
    background_count = 0

    for chunk_start in range(0, len(time_214), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(time_214))
        time_f_chunk = time_np[chunk_start:chunk_end]
        energy_chunk = energy_np[chunk_start:chunk_end]
        x_chunk = x_np[chunk_start:chunk_end]
        y_chunk = y_np[chunk_start:chunk_end]
        z_chunk = z_np[chunk_start:chunk_end]
        MC_f_chunk = MC_np[chunk_start:chunk_end]

        r = np.sqrt(x_chunk ** 2 + y_chunk ** 2)
        distances = np.sqrt((x_chunk[:, None] - x_chunk) ** 2 + (y_chunk[:, None] - y_chunk) ** 2 + (
                    z_chunk[:, None] - z_chunk) ** 2)

        #only events in this shell are considered(FV cut)
        valid_shell = (r <= r_fv_214) & (z_chunk <= h_fv_214) & (z_chunk >= -h_fv_214)
        
        # Apply other conditions only to valid events
        valid_indices_fv = np.where(valid_shell)[0]

        valid_energy_bi = (energy_chunk[valid_indices_fv] >= E_bi_low_214) & (
                    energy_chunk[valid_indices_fv] <= E_bi_up_214)
        
        #valid_indices_enrg_bi = np.where(valid_energy_bi)[0]

        valid_energy_po= (energy_chunk[valid_indices_fv] >= E_po_low_214) & (
                    energy_chunk[valid_indices_fv] <= E_po_up_214)
        
        #valid_indices_enrg_po = np.where(valid_energy_po)[0]
        

        # Calculate pairwise time differences efficiently
        time_diff = np.abs(time_f_chunk[valid_indices_fv][:, None] - time_f_chunk[valid_indices_fv])

        # Mask for valid time differences
        valid_time_diff_214 = (time_diff >= 0.0000002) & (time_diff <= t1_t2_214)

        # Convert MC_f truths to integers
        y_true_all = np.array(MC_f_chunk, dtype=int)

        # Initialize y_pred_all
        y_pred_all = np.zeros(len(y_true_all), dtype=int)

        # Exclude events with r > r_fv
        valid_events_bi_214 = valid_energy_bi & valid_time_diff_214

        # Apply other conditions only to valid events
        valid_indices_bi_214 = np.where(valid_events_bi_214)[0]

        # Exclude events with r > r_fv
        valid_events_po_214 = valid_energy_po & valid_time_diff_214

        # Apply other conditions only to valid events
        valid_indices_po_214 = np.where(valid_events_po_214)[0]

        # Energy condition for truth 1 (Bi)
        bi_condition_214 =  np.any(distances[valid_indices_bi_214][:, valid_indices_bi_214] < dist_214, axis=1)

        # Energy condition for truth 2 (Po)
        po_condition_214 = np.any(distances[valid_indices_po_214][:, valid_indices_po_214] < dist_214, axis=1)

        # Initialize y_pred_all with 0 (indicating unassigned)
        #y_pred_all = np.full(len(y_true_all), 0, dtype=int) 

        # Assign predictions based on conditions
        y_pred_all[valid_indices_bi_214[bi_condition_214]] = 1
        y_pred_all[valid_indices_po_214[po_condition_214]] = 2

        # Calculate TP, FP, TN, FN for the current chunk
        # Initialize counts for true positives, false positives, true negatives, and false negatives
        tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0

        tp = np.sum((y_true_all == 1) & (y_pred_all == 1)) + np.sum((y_true_all == 2) & (y_pred_all == 2))  # True positives: Both y_true and y_pred are not background
        fp = np.sum((y_true_all != 1) & (y_pred_all == 1)) + np.sum((y_true_all != 2) & (y_pred_all == 2))  # False positives: y_pred is not background but y_true is
        tn = np.sum((y_true_all == 0) & (y_pred_all == 0)) + np.sum((y_true_all == 3) & (y_pred_all == 3))  # True negatives: Both y_true and y_pred are background
        fn = np.sum((y_true_all != 0) & (y_pred_all == 0)) + np.sum((y_true_all != 3) & (y_pred_all == 3))  # False negatives: y_pred is background but y_true is not

        # Accumulate counts for all iterations
        tp_total += tp
        fp_total += fp
        tn_total += tn
        fn_total += fn
        # Update signal and background counts
        signal_count += tp_total
        background_count += fp_total
    
    # Calculate the average signal-to-noise ratio

    signal = signal_count
    background = background_count
    print("Signal:{}, Background:{}".format(signal, background))
    snr = signal / np.sqrt(signal + background)

    return snr


# Create the Optuna study and optimize the objective function
study_1 = optuna.create_study(direction='maximize')
study_1.optimize(objective_1, n_trials=100)

# Get the best parameters
best_params = study_1.best_params
best_r_fv_214 = best_params['r_fv_214']
best_h_fv_214 = best_params['h_fv_214']
best_t1_t2_214 = best_params['t1_t2_214']
best_E_bi_up_214 = best_params['E_bi_up_214']
best_E_bi_low_214 = best_params['E_bi_low_214']
best_E_po_up_214 = best_params['E_po_up_214']
best_E_po_low_214 = best_params['E_po_low_214']
best_dist_214 = best_params['dist_214']

fig_1 = plot_optimization_history(study_1)
plt.show()
# Save the image as a .jpg file
plt.savefig('classical_cuts/opt_hist_214.jpg', format='jpg')

def objective_2(trial):
    # Define the hyperparameters to tune
    r_fv_212 = trial.suggest_float('r_fv_212', 800, 1500)
    t1_t2_212 = trial.suggest_float('t1_t2_212', 0.0000002, 0.0000300)
    E_bi_up_212 = trial.suggest_float('E_bi_up_212', 1, 4.5)
    E_bi_low_212 = trial.suggest_float('E_bi_low_212', 0.05, 2.5)
    E_po_up_212 = trial.suggest_float('E_po_up_212', 0.8, 5)
    E_po_low_212 = trial.suggest_float('E_po_low_212', 0.1, 1.5)
    dist_212 = trial.suggest_float('dist_212', 450, 1200)
    h_fv_212 = trial.suggest_float('h_fv_212', 1000, 1500)

    # Use NumPy arrays for vectorized operations
    time_np = np.array(time_212)
    energy_np = np.array(energy_212)
    x_np = np.array(x_212)
    y_np = np.array(y_212)
    z_np = np.array(z_212)
    MC_np = np.array(MC_212)

    # Define chunk size for processing data
    chunk_size = 15000

    signal_count = 0
    background_count = 0

    for chunk_start in range(0, len(time_212), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(time_212))
        time_f_chunk = time_np[chunk_start:chunk_end]
        energy_chunk = energy_np[chunk_start:chunk_end]
        x_chunk = x_np[chunk_start:chunk_end]
        y_chunk = y_np[chunk_start:chunk_end]
        z_chunk = z_np[chunk_start:chunk_end]
        MC_f_chunk = MC_np[chunk_start:chunk_end]

        r = np.sqrt(x_chunk ** 2 + y_chunk ** 2)
        distances = np.sqrt((x_chunk[:, None] - x_chunk) ** 2 + (y_chunk[:, None] - y_chunk) ** 2 + (
                    z_chunk[:, None] - z_chunk) ** 2)

        #only events in this shell are considered(FV cut)
        valid_shell = (r <= r_fv_212) & (z_chunk <= h_fv_212) & (z_chunk >= -h_fv_212)
        
        # Apply other conditions only to valid events
        valid_indices_fv = np.where(valid_shell)[0]

        valid_energy_bi = (energy_chunk[valid_indices_fv] >= E_bi_low_212) & (
                    energy_chunk[valid_indices_fv] <= E_bi_up_212)
        
        #valid_indices_enrg_bi = np.where(valid_energy_bi)[0]

        valid_energy_po= (energy_chunk[valid_indices_fv] >= E_po_low_212) & (
                    energy_chunk[valid_indices_fv] <= E_po_up_212)
        
        #valid_indices_enrg_po = np.where(valid_energy_po)[0]
        

        # Calculate pairwise time differences efficiently
        time_diff = np.abs(time_f_chunk[valid_indices_fv][:, None] - time_f_chunk[valid_indices_fv])

        # Mask for valid time differences
        valid_time_diff_212 = (time_diff >= 0.0000002) & (time_diff <= t1_t2_212)

        # Convert MC_f truths to integers
        y_true_all = np.array(MC_f_chunk, dtype=int)

        # Initialize y_pred_all
        y_pred_all = np.zeros(len(y_true_all), dtype=int)

        # Exclude events with r > r_fv
        valid_events_bi_212 = valid_energy_bi & valid_time_diff_212

        # Apply other conditions only to valid events
        valid_indices_bi_212 = np.where(valid_events_bi_212)[0]

        # Exclude events with r > r_fv
        valid_events_po_212 = valid_energy_po & valid_time_diff_212

        # Apply other conditions only to valid events
        valid_indices_po_212 = np.where(valid_events_po_212)[0]

        # Energy condition for truth 1 (Bi)
        bi_condition_214 =  np.any(distances[valid_indices_bi_212][:, valid_indices_bi_212] < dist_212, axis=1)

        # Energy condition for truth 2 (Po)
        po_condition_214 = np.any(distances[valid_indices_po_212][:, valid_indices_po_212] < dist_212, axis=1)

        # Initialize y_pred_all with 0 (indicating unassigned)
        #y_pred_all = np.full(len(y_true_all), 0, dtype=int) 

        # Assign predictions based on conditions
        y_pred_all[valid_indices_bi_212[bi_condition_214]] = 4
        y_pred_all[valid_indices_po_212[po_condition_214]] = 5

        # Calculate TP, FP, TN, FN for the current chunk
        # Initialize counts for true positives, false positives, true negatives, and false negatives
        tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0

        tp = (np.sum((y_true_all == 4) & (y_pred_all == 4)) + np.sum((y_true_all == 5) & (y_pred_all == 5)))  # True positives: Both y_true and y_pred are not background
        fp = np.sum((y_true_all != 4) & (y_pred_all == 4)) + np.sum((y_true_all != 5) & (y_pred_all == 5))  # False positives: y_pred is not background but y_true is
        tn = np.sum((y_true_all == 0) & (y_pred_all == 0)) + np.sum((y_true_all == 3) & (y_pred_all == 3))  # True negatives: Both y_true and y_pred are background
        fn = np.sum((y_true_all != 0) & (y_pred_all == 0)) + np.sum((y_true_all != 3) & (y_pred_all == 3))  # False negatives: y_pred is background but y_true is not

        # Accumulate counts for all iterations
        tp_total += tp
        fp_total += fp
        tn_total += tn
        fn_total += fn
        # Update signal and background counts
        signal_count += tp_total
        background_count += fp_total
    
    # Calculate the average signal-to-noise ratio

    signal = signal_count
    background = background_count
    print("Signal:{}, Background:{}".format(signal, background))
    snr = signal / np.sqrt(signal + background)

    return snr

# Create the Optuna study and optimize the objective function
study_2 = optuna.create_study(direction='maximize')
study_2.optimize(objective_2, n_trials=120)

# Get the best parameters
best_params = study_2.best_params
best_r_fv_212 = best_params['r_fv_212']
best_h_fv_212 = best_params['h_fv_212']
best_t1_t2_212 = best_params['t1_t2_212']
best_E_bi_up_212 = best_params['E_bi_up_212']
best_E_bi_low_212 = best_params['E_bi_low_212']
best_E_po_up_212 = best_params['E_po_up_212']
best_E_po_low_212 = best_params['E_po_low_212']
best_dist_212 = best_params['dist_212']


print(f"Best r_fv_214 value: {best_r_fv_214}")
print(f"Best h_fv_214 value: {best_h_fv_214}")
print(f"Best t1_t2_214 value: {best_t1_t2_214}")
print(f"Best E_bi_up_214 value: {best_E_bi_up_214}")
print(f"Best E_bi_low_214 value: {best_E_bi_low_214}")
print(f"Best E_po_up_214 value: {best_E_po_up_214}")
print(f"Best E_po_low_214 value: {best_E_po_low_214}")
print(f"Best dist_212 value: {best_dist_214}")
print(f"Best r_fv_212 value: {best_r_fv_212}")
print(f"Best h_fv_212 value: {best_h_fv_212}")
print(f"Best t1_t2_212 value: {best_t1_t2_212}")
print(f"Best E_bi_up_212 value: {best_E_bi_up_212}")
print(f"Best E_bi_low_212 value: {best_E_bi_low_212}")
print(f"Best E_po_up_212 value: {best_E_po_up_212}")
print(f"Best E_po_low_212 value: {best_E_po_low_212}")
print(f"Best dist_212 value: {best_dist_212}")


with open('classical_cuts/best_parameters.txt', 'w') as f:
    f.write(f"Best parameters:\n")
    f.write(f"Best r_fv_214 value: {best_r_fv_214}\n")
    f.write(f"Best h_fv_214 value: {best_h_fv_214}\n")
    f.write(f"Best t1_t2_214 value: {best_t1_t2_214}\n")
    f.write(f"Best E_bi_up_214 value: {best_E_bi_up_214}\n")
    f.write(f"Best E_bi_low_214 value: {best_E_bi_low_214}\n")
    f.write(f"Best E_po_up_214 value: {best_E_po_up_214}\n")
    f.write(f"Best E_po_low_214 value: {best_E_po_low_214}\n")
    f.write(f"Best dist_214 value: {best_dist_214}\n")
    f.write(f"Best r_fv_212 value: {best_r_fv_212}\n")
    f.write(f"Best h_fv_up_212 value: {best_h_fv_212}\n")
    f.write(f"Best t1_t2_212 value: {best_t1_t2_212}\n")
    f.write(f"Best E_bi_up_212 value: {best_E_bi_up_212}\n")
    f.write(f"Best E_bi_low_212 value: {best_E_bi_low_212}\n")
    f.write(f"Best E_po_up_212 value: {best_E_po_up_212}\n")
    f.write(f"Best E_po_low_212 value: {best_E_po_low_212}\n")
    f.write(f"Best dist_212 value: {best_dist_212}\n")



fig_2 = plot_optimization_history(study_2)
plt.show()
plt.savefig('classical_cuts/opt_hist_212.jpg', format='jpg')