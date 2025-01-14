import os  # OPERATING SYSTEM OPERATIONS
import numpy as np  # NUMERICAL OPERATIONS
import matplotlib.pyplot as plt  # PLOTTING
from sklearn.decomposition import PCA  # DIMENSIONALITY REDUCTION
from scipy.linalg import svd  # SINGULAR VALUE DECOMPOSITION
import jax  # IMPORT JAX MODULE
import jax.numpy as jnp  # JAX NUMPY FOR GPU ACCELERATION
from flax import linen as nn  # FLAX NEURAL NETWORK MODULE
from flax.training import train_state  # FLAX TRAINING STATE
import optax  # OPTIMIZATION PACKAGE
from tqdm import tqdm  # PROGRESS BAR
from PIL import Image  # IMAGE PROCESSING


# LOAD IMAGES FROM FOLDER AND PROCESS THEM
def load_images_from_folder(folder_path, img_size=(200, 200)):  # IMAGE LOADING FUNCTION
    images = []  # LIST TO STORE IMAGES
    for filename in tqdm(os.listdir(folder_path), desc=f"Loading images from {folder_path}"):  # ITERATE FILES
        img_path = os.path.join(folder_path, filename)  # FULL PATH
        try:
            img = Image.open(img_path).convert("L")  # OPEN AND CONVERT TO GRAYSCALE
            img = img.resize(img_size)  # RESIZE TO 200x200
            images.append(np.array(img))  # APPEND IMAGE AS NUMPY ARRAY
        except Exception as e:  # HANDLE ERRORS
            print(f"Could not process {filename}: {e}")  # ERROR MESSAGE
    return np.array(images)  # RETURN NUMPY ARRAY OF IMAGES

# BASIC DMD IMPLEMENTATION
def dmd(X1, X2, r=None):  # PERFORM DYNAMIC MODE DECOMPOSITION
    U, Sigma, VT = svd(X1, full_matrices=False)  # SVD ON X1 TO GET U, SIGMA, AND VT
    if r is not None:  # IF REDUCED DIMENSION IS SPECIFIED
        U = U[:, :r]  # KEEP ONLY FIRST r COLUMNS OF U
        Sigma = np.diag(Sigma[:r])  # KEEP ONLY FIRST r SINGULAR VALUES
        VT = VT[:r, :]  # KEEP ONLY FIRST r ROWS OF VT
    A_tilde = U.T.conj() @ X2 @ VT.T.conj() @ np.linalg.inv(Sigma)  # APPROXIMATE LINEAR OPERATOR
    eigenvalues, W = np.linalg.eig(A_tilde)  # COMPUTE EIGENVALUES AND EIGENVECTORS OF A_TILDE
    Phi = X2 @ VT.T.conj() @ np.linalg.inv(Sigma) @ W  # PROJECT EIGENVECTORS BACK TO ORIGINAL SPACE TO GET DYNAMIC MODES
    return eigenvalues, Phi  # RETURN EIGENVALUES AND MODES

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

# MAIN FUNCTION
if __name__ == "__main__":  # ENTRY POINT FOR SCRIPT
    # LOAD AND PROCESS IMAGES
    emotion_images = load_images_from_folder("task-emotion")  # LOAD IMAGES FROM task-emotion FOLDER
    rest_images = load_images_from_folder("task-rest")  # LOAD IMAGES FROM task-rest FOLDER

    # SAVE PROCESSED IMAGES AS NPZ FILES
    np.savez("task-emotion.npz", arr_0=emotion_images)  # SAVE emotion_images AS task-emotion.npz
    np.savez("task-rest.npz", arr_0=rest_images)  # SAVE rest_images AS task-rest.npz

    # LOAD DATASETS
    data = np.load("task-emotion.npz")  # LOAD NPZ FILE CONTAINING DATA
    X = data["arr_0"]  # EXTRACT FIRST ARRAY FROM NPZ FILE

    # RESHAPE TO 2D MATRICES FOR DMD
    X1 = X[:, :-1].reshape(X.shape[0] * X.shape[1], -1)  # FLATTEN IMAGES INTO COLUMNS FOR X1
    X2 = X[:, 1:].reshape(X.shape[0] * X.shape[1], -1)  # FLATTEN IMAGES INTO COLUMNS FOR X2

    # RUN BASIC DMD
    eigenvalues, Phi = dmd(X1, X2, r=10)  # RUN BASIC DMD ON DATA
    print("Eigenvalues:", eigenvalues)  # PRINT EIGENVALUES FROM DMD

    # INITIALIZE AUTOENCODER AND TRAIN STATE
    rng = jax.random.PRNGKey(0)  # INITIALIZE RANDOM KEY FOR JAX
    autoencoder = Autoencoder(latent_dim=10, output_dim=200)  # INITIALIZE AUTOENCODER MODULE
    state = create_train_state(rng, autoencoder, learning_rate=1e-3)  # CREATE TRAINING STATE WITH LEARNING RATE

    # TRAINING LOOP
    for epoch in tqdm(range(100), desc="Training Autoencoder"):  # TRAINING LOOP WITH PROGRESS BAR
        state, loss = train_step(state, jnp.array(X1))  # PERFORM ONE EPOCH OF TRAINING
        if epoch % 10 == 0:  # PRINT LOSS EVERY 10 EPOCHS
            print(f"Epoch {epoch}, Loss: {loss:.4f}")  # OUTPUT EPOCH NUMBER AND LOSS VALUE

    # RECONSTRUCT DATA USING FINAL MODEL
    reconstructed = state.apply_fn({"params": state.params}, jnp.array(X1))  # RECONSTRUCT DATA USING FINAL MODEL

    # PLOTTING RESULTS
    plt.figure(figsize=(12, 6))  # CREATE FIGURE FOR PLOTTING
    plt.subplot(1, 2, 1)  # ADD FIRST SUBPLOT
    plt.title("Original Data")  # TITLE FOR ORIGINAL DATA PLOT
    plt.imshow(X1[:, :100], aspect="auto")  # DISPLAY ORIGINAL DATA AS IMAGE
    plt.subplot(1, 2, 2)  # ADD SECOND SUBPLOT
    plt.title("Reconstructed Data")  # TITLE FOR RECONSTRUCTED DATA PLOT
    plt.imshow(reconstructed[:, :100], aspect="auto")  # DISPLAY RECONSTRUCTED DATA AS IMAGE
    plt.show()  # SHOW PLOTS

