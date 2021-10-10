import os
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from callbacks import *
from dataio import prepare_dataset
from models import create_model
import hyperparameters


import warnings
warnings.filterwarnings("ignore")


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




Base_alpha = 32
Base_N_Heads = 256**2
use_bias_sparse = False
N_Heads = [1**2, 2**2, 4**2, 8**2, 16**2, 32**2, 64**2, 128**2, 256**2]


Random_Image = perlin_noise((256, 256), 8)


Base_Model_N_Parameters = calculate_parameters(hyperparameters.INPUT_COORDINATES,
											   hyperparameters.BASE_HIDDEN_UNITS,
											   hyperparameters.BASE_N_HLAYERS,
											   Base_N_Heads,
											   Base_alpha,
											   use_bias_sparse)




Total_Historys = []
for counter_Image, Image in enumerate(Random_Image):
    print(40*"*", f"Octave : {2 ** counter_Image}", 40*"*")

    Head_Historys = []
    for counter, n_heads in enumerate(N_Heads):
        print(30*"*", f"Head : {n_heads}", 30*"*")
        learningrate_scheduler = LearningRateScheduler()

        dataset = prepare_dataset(Image, n_heads)

        hidden_units, alpha_sparse = calculate_hyperparameter(Base_Model_N_Parameters,
        													  hyperparameters.INPUT_COORDINATES,
        													  hyperparameters.BASE_HIDDEN_UNITS,
        													  hyperparameters.BASE_N_HLAYERS,
        													  n_heads,
        													  Base_alpha,
        													  use_bias_sparse)
        MH_RELU_INR = create_model("multi-head",
                                   n_heads,
                                   hidden_units,
                                   alpha_sparse,
                                   use_bias_sparse)
        print("Number of Parameters : ", MH_RELU_INR.count_params())


        optimizer = tf.keras.optimizers.Adam(hyperparameters.LEARNING_RATE)
        loss_fn = tf.keras.losses.MeanSquaredError()
        MH_RELU_INR.compile(optimizer=optimizer, loss=loss_fn)

        #print(MH_RELU_INR.summary())

        save_result_callback = CustomSaver(Image, n_heads, hyperparameters.STEP_SHOW, loss_fn)

        start = time.time()
        history = MH_RELU_INR.fit(dataset,
                                  epochs=hyperparameters.EPOCHS,
                                  callbacks=[learningrate_scheduler, save_result_callback],
                                  verbose=0) # if you want to see train progress change verbose to 1
        print(f"Total Time : {time.time() - start}")

        Head_Historys.append(history.history["loss"])




    Total_Historys.append(Head_Historys)








##################################################### Plot Result ####################################################
min_frequency_bias = PSNR(np.min(np.array(Total_Historys), axis=-1))

linestyle = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted", "-."]
marker = ['o', 'v', 'p', 'P', '*', 'x', 'D', '4', 's']

for counter in range(len(marker)):
    plt.plot(min_frequency_bias[:, counter], marker=marker[counter], markersize=5, linestyle=linestyle[counter])

plt.grid()
plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=11)
plt.tick_params(axis='both', which='minor', labelsize=8)


plt.xlim([-0.1, 7.1])
plt.ylabel("PSNR(dB)", fontsize=10, fontweight="bold")
plt.xlabel(r'$\mathbf{log_{2}{(\:Octave\:)}}}$', fontsize=10)

plt.legend([r'$\mathbf{N\:of\:\:Heads : 1 \times 1}$',
            r'$\mathbf{N\:of\:\:Heads : 2 \times 2}$',
            r'$\mathbf{N\:of\:\:Heads : 4 \times 4}$',
            r'$\mathbf{N\:of\:\:Heads : 8 \times 8}$',
            r'$\mathbf{N\:of\:\:Heads : 16 \times 16}$',
            r'$\mathbf{N\:of\:\:Heads : 32 \times 32}$',
            r'$\mathbf{N\:of\:\:Heads : 64 \times 64}$',
            r'$\mathbf{N\:of\:\:Heads : 128 \times 128}$',
            r'$\mathbf{N\:of\:\:Heads : 256 \times 256}$'], fontsize=6.2, bbox_to_anchor=(0.32,0.93))

plt.savefig("results/Frequency_Bias.pdf", dpi=500, bbox_inches='tight', pad_inches=0)
plt.show()
######################################################################################################################