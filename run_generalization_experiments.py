import os
import time

import tensorflow as tf
import numpy as np
import skimage
import matplotlib.pyplot as plt
import cv2

from utils import *
from callbacks import *
from dataio import prepare_dataset
from models import create_model
import hyperparameters


import warnings
warnings.filterwarnings("ignore")


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path",
                default="cameraman",
                type=str,
                help="path of input image")

ap.add_argument("-bh", "--base_nheads",
                default=256,
                type=int,
                help="root number of head for base multi-head network(for fair comparison)")

ap.add_argument("-ba", "--base_alpha",
                default=32,
                type=int,
                help="alpha parameters for base multi-head network(for fair comparison)")

ap.add_argument("-ub", "--use_bias_sparse",
                default=False,
                type=bool,
                help="Use the bias for the head part of the multi-head network")

args = vars(ap.parse_args())




image_path = args["image_path"]
Base_N_Heads = args["base_nheads"] ** 2
Base_alpha = args["base_alpha"]
use_bias_sparse = args["use_bias_sparse"]

N_Heads = [1**2, 2**2, 4**2, 8**2, 16**2, 32**2, 64**2, 128**2, 256**2]




if image_path == "cameraman":
    Major_Image = skimage.data.camera()
    Major_Image = Major_Image.astype(np.float32) / 255.
    Minor_Image = cv2.resize(Major_Image, (256, 256))
else:
    Major_Image = cv2.imread(image_path)
    Major_Image = Major_Image.astype(np.float32) / 255.
    if len(Major_Image.shape) == 3:
        Major_Image = cv2.cvtColor(Major_Image, cv2.COLOR_BGR2GRAY)

    Major_Image = cv2.resize(Major_Image, (512, 512))
    Minor_Image = cv2.resize(Major_Image, (256, 256))



Base_Model_N_Parameters = calculate_parameters(hyperparameters.INPUT_COORDINATES,
											   hyperparameters.BASE_HIDDEN_UNITS,
											   hyperparameters.BASE_N_HLAYERS,
											   Base_N_Heads,
											   Base_alpha,
											   use_bias_sparse)



Save_Result_Callbacks_Minor = []
Save_Result_Callbacks_Major = []


Head_Historys = []
for counter, n_heads in enumerate(N_Heads):
    print(30*"*", f"Head : {n_heads}", 30*"*")
    learningrate_scheduler = LearningRateScheduler()

    dataset = prepare_dataset(Minor_Image, n_heads)

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

    save_result_callback_minor = CustomSaver(Minor_Image, n_heads, hyperparameters.STEP_SHOW, loss_fn)
    save_result_callback_major = CustomSaver(Major_Image, n_heads, hyperparameters.STEP_SHOW, loss_fn)

    start = time.time()
    history = MH_RELU_INR.fit(dataset,
                              epochs=hyperparameters.EPOCHS,
                              callbacks=[learningrate_scheduler, save_result_callback_minor, save_result_callback_major],
                              verbose=0) # if you want to see train progress change verbose to 1
    print(f"Total Time : {time.time() - start}")

    Head_Historys.append(history.history["loss"])
    Save_Result_Callbacks_Minor.append(save_result_callback_minor)
    Save_Result_Callbacks_Major.append(save_result_callback_major)



Total_Historys.append(Head_Historys)


psnr_minor = []
psnr_major = []
for counter in range(len(Save_Result_Callbacks_Minor)):
    psnr_minor.append(Save_Result_Callbacks_Minor[counter].psnr)
    psnr_major.append(Save_Result_Callbacks_Major[counter].psnr)
psnr_minor = np.array(psnr_minor)
psnr_major = np.array(psnr_major)





##################################################### Plot Result ####################################################
plt.plot(psnr_minor.T[-1,:], color='#11e3e3', marker='o')
plt.plot(psnr_major.T[-1,:], color='#ff8aac', marker='o')

plt.grid()

plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=11)
plt.tick_params(axis='both', which='minor', labelsize=8)

plt.xlim([-0.1, 8.1])
plt.ylabel("PSNR(dB)", fontsize=10, fontweight="bold")
plt.xlabel(r'$\mathbf{log_{2}{(\sqrt[2]{Number\:of\:Heads})}}$', fontsize=10)

max_major = max(psnr_major.T[-1,:])
index_max_major = np.argmax(psnr_major.T[-1,:])
plt.scatter(index_max_major, max_major, s=100, facecolors='none', edgecolors='r')
plt.text(index_max_major - 0.5, max_major + 7, 'Best Generalization', style='italic', fontsize=8, fontweight="bold")

plt.legend(["Train", "Test"], fontsize=12)

plt.savefig("results/Generalization_log.pdf", dpi=500, bbox_inches='tight', pad_inches=0)
plt.show()
######################################################################################################################