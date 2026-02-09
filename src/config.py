N_INTERACTIONS = 3
F_DIM = 64

# Radial basis function (RBF) expansion parameters
RBF_CENTERS = [i * 0.1 for i in range(0, 301)]  
RBF_GAMMA = 10.0

# Activation function (Shifted Softplus)
ACTIVATION = "ssp"

POOLING = "sum"
LR = 1e-3
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 32
SEED = 42
