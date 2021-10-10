import tensorflow as tf
import numpy as np


from dataio import prepare_dataset
from utils import PSNR
import hyperparameters




def LearningRateScheduler():
	def scheduler(epoch, lr):
	    if epoch < hyperparameters.LEARNING_RATE_DECAY_STRATPOINT:
	        return lr
	    else:
	        if epoch % hyperparameters.LEARNING_RATE_DECAY_STEP == 0:
	            lr = lr * tf.math.exp(hyperparameters.LEARNING_RATE_DECAY_PARAMETERS)
	    return lr
	return tf.keras.callbacks.LearningRateScheduler(scheduler)





class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self, Image, n_heads, step_show, loss_fn):
        super(CustomSaver, self).__init__()
        self.Image = Image
        self.s_n_heads = int(np.sqrt(n_heads))
        self.step_show = step_show
        self.loss_fn = loss_fn
        self.image_shape = self.Image.shape
        

        self.data = prepare_dataset(self.Image, n_heads)
        self.images = []
        self.psnr = []
        
    def image_reconstruct(self):
        predicted = self.model.predict(self.data)
        predicted = predicted.T.reshape((predicted.shape[1], int(self.image_shape[0]/self.s_n_heads), int(self.image_shape[1]/self.s_n_heads)))
        part_size = [int(self.image_shape[0]/self.s_n_heads), int(self.image_shape[0]/self.s_n_heads)]
        
        Image_reconstructed = np.zeros(self.image_shape)
        for counter1 in range(self.s_n_heads):
            for counter2 in range(self.s_n_heads):
                Image_reconstructed[counter1 * part_size[0]: (counter1 + 1) * part_size[0], counter2 * part_size[1]: (counter2 + 1) * part_size[1]] = \
                predicted[counter1 * self.s_n_heads + counter2]
        return Image_reconstructed
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.step_show == 0:
            self.images.append(self.image_reconstruct())
            loss_value = self.loss_fn(self.images[-1], self.Image)
            self.psnr.append(PSNR(loss_value))
            print(f"Epoch {epoch:4d} : loss {loss_value:.10f} , PSNR {self.psnr[-1]:.5f}")