import numpy as np  # NUMERICAL OPERATIONS
import matplotlib.pyplot as plt  # PLOTTING
from sklearn.decomposition import PCA  # PRINCIPAL COMPONENT ANALYSIS
from scipy.linalg import svd  # SINGULAR VALUE DECOMPOSITION
import jax.numpy as jnp  # JAX NUMPY FOR GPU ACCELERATION
from flax import linen as nn  # FLAX FOR NEURAL NETWORKS
from flax.training import train_state  # FLAX TRAINING STATE
import optax  # OPTIMIZATION PACKAGE
from tqdm import tqdm  # PROGRESS BAR

# BASIC DMD IMPLEMENTATION
def dmd(X1, X2, r=None):  # PERFORM DYNAMIC MODE DECOMPOSITION
    U, Sigma, VT = svd(X1, full_matrices=False)  # SVD ON X1 TO GET U, SIGMA, AND VT
    if r is not None:  # IF REDUCED DIMENSION IS SPECIFIED
        U = U[:, :r]  # KEEP ONLY FIRST r COLUMNS OF U
        Sigma = np.diag(Sigma[:r])  # KEEP ONLY FIRST r SINGULAR VALUES
        VT = VT[:r, :]  # KEEP ONLY FIRST r ROWS OF VT
    A_tilde = U.T.conj() @ X2 @ VT.T.conj() @ np.linalg.inv(Sigma)  # APPROXIMATE LINEAR OPERATOR, TRANSFORMING INPUTS VIA SVD
    eigenvalues, W = np.linalg.eig(A_tilde)  # COMPUTE EIGENVALUES AND EIGENVECTORS OF A_TILDE
    Phi = X2 @ VT.T.conj() @ np.linalg.inv(Sigma) @ W  # PROJECT EIGENVECTORS BACK TO ORIGINAL SPACE TO GET DYNAMIC MODES
    return eigenvalues, Phi  # RETURN EIGENVALUES AND MODES

# EXTENDED DMD WITH NONLINEAR BASIS
def extended_dmd(X1, X2, basis_func, r=None):  # DMD WITH NONLINEAR TRANSFORMATION
    X1_ext = basis_func(X1)  # APPLY BASIS FUNCTION TO EXTEND X1
    X2_ext = basis_func(X2)  # APPLY BASIS FUNCTION TO EXTEND X2
    return dmd(X1_ext, X2_ext, r)  # RETURN RESULTS FROM DMD ON EXTENDED DATA

# EXAMPLE NONLINEAR BASIS FUNCTION
def quadratic_basis(X):  # NONLINEAR QUADRATIC BASIS
    return np.vstack([X, X**2])  # STACK ORIGINAL DATA AND SQUARED TERMS TO FORM NONLINEAR BASIS

# ENCODER MODULE USING FLAX
class Encoder(nn.Module):  # DEFINE ENCODER MODULE
    latent_dim: int  # LATENT DIMENSION SIZE SPECIFIED DURING INITIALIZATION
    @nn.compact
    def __call__(self, x):  # FORWARD PASS
        x = nn.Dense(128)(x)  # APPLY DENSE LAYER WITH 128 UNITS
        x = nn.relu(x)  # APPLY RELU ACTIVATION FUNCTION
        x = nn.Dense(self.latent_dim)(x)  # APPLY DENSE LAYER TO REDUCE TO LATENT DIMENSION
        return x  # RETURN ENCODED VECTOR

# DECODER MODULE USING FLAX
class Decoder(nn.Module):  # DEFINE DECODER MODULE
    output_dim: int  # OUTPUT DIMENSION SIZE SPECIFIED DURING INITIALIZATION
    @nn.compact
    def __call__(self, z):  # FORWARD PASS
        z = nn.Dense(128)(z)  # APPLY DENSE LAYER WITH 128 UNITS
        z = nn.relu(z)  # APPLY RELU ACTIVATION FUNCTION
        z = nn.Dense(self.output_dim)(z)  # APPLY FINAL DENSE LAYER TO OUTPUT RECONSTRUCTION
        return z  # RETURN DECODED VECTOR

# AUTOENCODER MODULE
class Autoencoder(nn.Module):  # DEFINE AUTOENCODER MODULE
    latent_dim: int  # LATENT DIMENSION SIZE
    output_dim: int  # OUTPUT DIMENSION SIZE
    def setup(self):  # INITIAL SETUP FOR ENCODER AND DECODER
        self.encoder = Encoder(self.latent_dim)  # INITIALIZE ENCODER MODULE
        self.decoder = Decoder(self.output_dim)  # INITIALIZE DECODER MODULE
    def __call__(self, x):  # FORWARD PASS THROUGH AUTOENCODER
        z = self.encoder(x)  # ENCODE INPUT INTO LATENT SPACE
        return self.decoder(z)  # DECODE BACK TO ORIGINAL DIMENSION

# CREATE TRAIN STATE
def create_train_state(rng, model, learning_rate):  # INITIALIZE TRAINING STATE WITH RANDOM KEY
    params = model.init(rng, jnp.ones([1, 200]))["params"]  # INITIALIZE MODEL PARAMETERS WITH SAMPLE INPUT
    tx = optax.adam(learning_rate)  # DEFINE ADAM OPTIMIZER
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)  # RETURN TRAINING STATE OBJECT

# TRAINING STEP
def train_step(state, batch):  # PERFORM ONE STEP OF TRAINING
    def loss_fn(params):  # DEFINE LOSS FUNCTION
        reconstructed = state.apply_fn({"params": params}, batch)  # RECONSTRUCT BATCH USING AUTOENCODER
        loss = jnp.mean((batch - reconstructed) ** 2)  # COMPUTE MEAN SQUARED ERROR LOSS
        return loss  # RETURN LOSS VALUE
    grad_fn = jax.value_and_grad(loss_fn)  # COMPUTE GRADIENTS OF LOSS FUNCTION
    loss, grads = grad_fn(state.params)  # APPLY GRADIENTS TO STATE PARAMETERS
    new_state = state.apply_gradients(grads=grads)  # UPDATE TRAINING STATE WITH NEW PARAMETERS
    return new_state, loss  # RETURN UPDATED STATE AND LOSS

# =====     MAIN FUNCTION       ================
if __name__ == "__main__":  # ENTRY POINT FOR SCRIPT
    data = np.load("task-emotion.npz")  # LOAD NPZ FILE CONTAINING DATA
    X = data["arr_0"]  # EXTRACT FIRST ARRAY FROM NPZ FILE
    X1 = X[:, :-1]  # SPLIT DATA INTO X1 (ALL BUT LAST COLUMN)
    X2 = X[:, 1:]  # SPLIT DATA INTO X2 (ALL BUT FIRST COLUMN)

    eigenvalues, Phi = dmd(X1, X2, r=10)  # RUN BASIC DMD ON DATA
    print("Eigenvalues:", eigenvalues)  # PRINT EIGENVALUES FROM DMD

    rng = jax.random.PRNGKey(0)  # INITIALIZE RANDOM KEY FOR JAX
    autoencoder = Autoencoder(latent_dim=10, output_dim=200)  # INITIALIZE AUTOENCODER MODULE
    state = create_train_state(rng, autoencoder, learning_rate=1e-3)  # CREATE TRAINING STATE WITH LEARNING RATE

    for epoch in tqdm(range(100), desc="Training Autoencoder"):  # TRAINING LOOP WITH PROGRESS BAR
        state, loss = train_step(state, jnp.array(X1))  # PERFORM ONE EPOCH OF TRAINING
        if epoch % 10 == 0:  # PRINT LOSS EVERY 10 EPOCHS
            print(f"Epoch {epoch}, Loss: {loss:.4f}")  # OUTPUT EPOCH NUMBER AND LOSS VALUE

    reconstructed = state.apply_fn({"params": state.params}, jnp.array(X1))  # RECONSTRUCT DATA USING FINAL MODEL

    plt.figure(figsize=(12, 6))  # CREATE FIGURE FOR PLOTTING
    plt.subplot(1, 2, 1)  # ADD FIRST SUBPLOT
    plt.title("Original Data")  # TITLE FOR ORIGINAL DATA PLOT
    plt.imshow(X1[:, :100], aspect="auto")  # DISPLAY ORIGINAL DATA AS IMAGE
    plt.subplot(1, 2, 2)  # ADD SECOND SUBPLOT
    plt.title("Reconstructed Data")  # TITLE FOR RECONSTRUCTED DATA PLOT
    plt.imshow(reconstructed[:, :100], aspect="auto")  # DISPLAY RECONSTRUCTED DATA AS IMAGE
    plt.show()  # SHOW PLOTS
