#Model Config
INPUT_COORDINATES = 2
BASE_N_HLAYERS = 4
BASE_HIDDEN_UNITS = 256



# Training Config
EPOCHS = 2000
STEP_SHOW = 100


# Optimizer Config
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY_PARAMETERS = -0.1
LEARNING_RATE_DECAY_STRATPOINT = 50
LEARNING_RATE_DECAY_STEP = 50




#SIREN Config
OMEGA_0 = 30

#Fourier Feature Config
SCALE = 10
FOURIER_UNITS = 128