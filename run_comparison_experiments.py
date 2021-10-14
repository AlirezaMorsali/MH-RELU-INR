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




Base_alpha = 256
Base_N_Heads = 64**2
use_bias_sparse = True
n_heads = 64**2




Major_Image = skimage.data.camera()
Major_Image = Major_Image.astype(np.float32) / 255.
Minor_Image = cv2.resize(Major_Image, (256, 256))



Base_Model_N_Parameters = calculate_parameters(hyperparameters.INPUT_COORDINATES,
                                               hyperparameters.BASE_HIDDEN_UNITS,
                                               hyperparameters.BASE_N_HLAYERS,
                                               Base_N_Heads,
                                               Base_alpha,
                                               use_bias_sparse)



Save_Result_Callbacks_Minor = []
Save_Result_Callbacks_Major = []



###################################################  Multi-Head Model ####################################################
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

Save_Result_Callbacks_Minor.append(save_result_callback_minor)
Save_Result_Callbacks_Major.append(save_result_callback_major)
##########################################################################################################################



#####################################################  SIREN Model #######################################################
print(30*"*", "  SIREN  ", 30*"*")
learningrate_scheduler = LearningRateScheduler()
dataset = prepare_dataset(Minor_Image, 1)

hidden_units, _ = calculate_hyperparameter(Base_Model_N_Parameters,
                                           hyperparameters.INPUT_COORDINATES,
                                           hyperparameters.BASE_HIDDEN_UNITS,
                                           hyperparameters.BASE_N_HLAYERS,
                                           1,
                                           Base_alpha,
                                           use_bias_sparse)    

siren_model = create_model("siren",
                           1,
                           hidden_units,
                           None,
                           None)
print("Number of Parameters : ", siren_model.count_params())

optimizer = tf.keras.optimizers.Adam(hyperparameters.LEARNING_RATE)
loss = tf.keras.losses.MeanSquaredError()
siren_model.compile(optimizer=optimizer, loss=loss)

save_result_callback_minor = CustomSaver(Minor_Image, 1, hyperparameters.STEP_SHOW, loss_fn)
save_result_callback_major = CustomSaver(Major_Image, 1, hyperparameters.STEP_SHOW, loss_fn)

start = time.time()
history = siren_model.fit(dataset,
                          epochs=hyperparameters.EPOCHS,
                          callbacks=[learningrate_scheduler, save_result_callback_minor, save_result_callback_major],
                          verbose=0) # if you want to see train progress change verbose to 1
print(f"Total Time : {time.time() - start}")


Save_Result_Callbacks_Minor.append(save_result_callback_minor)
Save_Result_Callbacks_Major.append(save_result_callback_major)
###########################################################################################################################



##################################################### Fourier Model #######################################################
print(30*"*", "  Fourier  ", 30*"*")
learningrate_scheduler = LearningRateScheduler()
dataset = prepare_dataset(Minor_Image, 1)



hidden_units, _ = calculate_hyperparameter(Base_Model_N_Parameters,
                                           hyperparameters.INPUT_COORDINATES * hyperparameters.FOURIER_UNITS,
                                           hyperparameters.BASE_HIDDEN_UNITS,
                                           hyperparameters.BASE_N_HLAYERS,
                                           1,
                                           Base_alpha,
                                           use_bias_sparse)    


fourier_model = create_model("fourier",
                             1,
                             hidden_units,
                             None,
                             None)  
print("Number of Parameters : ", fourier_model.count_params()) 

optimizer = tf.keras.optimizers.Adam(hyperparameters.LEARNING_RATE)
loss = tf.keras.losses.MeanSquaredError()
fourier_model.compile(optimizer=optimizer, loss=loss)

save_result_callback_minor = CustomSaver(Minor_Image, 1, hyperparameters.STEP_SHOW, loss_fn)
save_result_callback_major = CustomSaver(Major_Image, 1, hyperparameters.STEP_SHOW, loss_fn)

start = time.time()
history = fourier_model.fit(dataset,
                            epochs=hyperparameters.EPOCHS,
                            callbacks=[learningrate_scheduler, save_result_callback_minor, save_result_callback_major],
                            verbose=0) # if you want to see train progress change verbose to 1
print(f"Total Time : {time.time() - start}")


Save_Result_Callbacks_Minor.append(save_result_callback_minor)
Save_Result_Callbacks_Major.append(save_result_callback_major)
############################################################################################################################



psnr_minor = []
psnr_major = []
for counter in range(len(Save_Result_Callbacks_Minor)):
    psnr_minor.append(Save_Result_Callbacks_Minor[counter].psnr)
    psnr_major.append(Save_Result_Callbacks_Major[counter].psnr)
psnr_minor = np.array(psnr_minor)
psnr_major = np.array(psnr_major)





####################################################### Plot Result ########################################################
linestyle = ["-", "--", "-."]


plt.figure()

for counter in range(len(linestyle)):
    plt.plot(psnr_major[counter, :], linestyle=linestyle[counter], linewidth=2)

plt.grid()
plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=11)
plt.tick_params(axis='both', which='minor', labelsize=8)

plt.xticks(np.arange(0, int(hyperparameters.EPOCHS / hyperparameters.STEP_SHOW) + 1, int(hyperparameters.EPOCHS / hyperparameters.STEP_SHOW) / 10),
           np.arange(0, hyperparameters.EPOCHS + 1, int(hyperparameters.EPOCHS / 10)))

plt.xlim([-1, 81])
plt.ylabel("PSNR(dB)", fontsize=10, fontweight="bold")
plt.xlabel('Epoch', fontsize=10, fontweight="bold")

plt.legend([r'Multi Head($64^2\:heads$)', "SIREN", "Fourier Feature"], fontsize=11)

plt.savefig(f"results/comparison(alpha={Base_alpha}).pdf", dpi=500, bbox_inches='tight', pad_inches=0)
plt.show()


plt.figure()
buff = Save_Result_Callbacks_Minor[0].images[-1]
plt.imshow((buff - buff.min()) / (buff.max() - buff.min()))
plt.savefig(f"results/comparison_MultiHead_Image(alpha={Base_alpha}).pdf", dpi=500, bbox_inches='tight', pad_inches=0)
plt.show()

plt.figure()
buff = Save_Result_Callbacks_Minor[1].images[-1]
plt.imshow((buff - buff.min()) / (buff.max() - buff.min()))
plt.savefig(f"results/comparison_SIREN_Image(alpha={Base_alpha}).pdf", dpi=500, bbox_inches='tight', pad_inches=0)
plt.show()

plt.figure()
buff = Save_Result_Callbacks_Minor[2].images[-1]
plt.imshow((buff - buff.min()) / (buff.max() - buff.min()))
plt.savefig(f"results/comparison_FourierFeature_Image(alpha={Base_alpha}).pdf", dpi=500, bbox_inches='tight', pad_inches=0)
plt.show()
############################################################################################################################
