import numpy as np
import matplotlib.pyplot as plt
import os

def load_record(record_model_type):

    record_model_type

    root_path = os.path.join(os.getcwd(), 'maze_example', 'savefile')

    record_data = np.load(os.path.join(root_path, record_model_type, 'reward_plot.npy'))

    return record_data

def main_plot():

    seed = 7777

    # sac_data = load_record('sac')

    sac_g20_data = load_record('sac_g20')

    # redq_data = load_record('redq')

    redq_v2_data = load_record('redq_v2')

    fig = plt.figure(figsize=(10, 7))
    
    ax1 = fig.add_subplot(1, 1, 1)
    
    # ax1.plot(sac_data[:, 2], label="Soft Actor Critic")
    
    ax1.plot(sac_g20_data[:, 2], label="Soft Actor Critic with G = 20")
    
    # ax1.plot(redq_data[:, 2], label="REDQ")
    
    ax1.plot(redq_v2_data[:, 2], label="REDQ V2")
    
    ax1.set_ylabel("cumulative rewards")
    
    ax1.set_xlabel("episode")

    ax1.set_title(f"maze example : seed {seed:d}")
    
    ax1.set_xlim([0, 80])
    
    ax1.set_ylim([0, 400])

    ax1.grid()

    ax1.legend()

    # plt.show()

    img_path = os.path.join(os.getcwd(), 'img')

    plt.savefig(os.path.join(img_path, "comparison_maze.png"))


if __name__ == '__main__':

    main_plot()
