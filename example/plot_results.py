import numpy as np
import matplotlib.pyplot as plt
import os

def load_record(record_model_type):

    record_model_type

    root_path = os.path.join(os.getcwd(), 'example', 'savefile')

    record_data = np.load(os.path.join(root_path, record_model_type, 'reward_plot.npy'))

    return record_data

def main_plot():

    sac_data = load_record('sac')

    redq_data = load_record('redq')

    fig = plt.figure()
    
    ax1 = fig.add_subplot(1, 1, 1)
    
    ax1.plot(sac_data[:, 0], label="Soft Actor Critic")
    
    ax1.plot(redq_data[:, 0], label="REDQ")
    
    ax1.set_ylabel("performance")
    
    ax1.set_xlabel("episode")
    
    ax1.set_xlim([0, 50])

    ax1.grid()

    ax1.legend()

    plt.show()




if __name__ == '__main__':

    main_plot()
