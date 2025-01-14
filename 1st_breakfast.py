import os  # OPERATING SYSTEM OPERATIONS
import numpy as np  # NUMERICAL OPERATIONS
from PIL import Image  # IMAGE PROCESSING
import torch  # PYTORCH FOR MACHINE LEARNING
import torch.nn as nn  # NEURAL NETWORK MODULES
import torch.optim as optim  # OPTIMIZATION FUNCTIONS
from scipy.linalg import svd  # SINGULAR VALUE DECOMPOSITION

# FUNCTION TO LOAD AND PROCESS IMAGES FROM A FOLDER
def load_images(folder):  # LOAD IMAGES FUNCTION
    images = []  # LIST TO STORE IMAGES
    for filename in os.listdir(folder):  # ITERATE OVER FILES IN FOLDER
        img = Image.open(os.path.join(folder, filename)).convert('L')  # OPEN AND CONVERT IMAGE TO GRAYSCALE
        img = img.resize((200, 200))  # RESIZE IMAGE TO 200x200
        images.append(np.array(img).flatten())  # FLATTEN IMAGE AND APPEND TO LIST
    return np.array(images)  # RETURN ARRAY OF IMAGES

# FUNCTION TO PERFORM DYNAMIC MODE DECOMPOSITION
def dmd(X1, X2, r=None):  # DMD FUNCTION
    U, Sigma, VT = svd(X1, full_matrices=False)  # SVD ON X1 TO GET U, SIGMA, AND VT
    if r is not None:  # CHECK IF REDUCED DIMENSION IS SPECIFIED
        U = U[:, :r]  # TRUNCATE U TO FIRST r COLUMNS
        Sigma = np.diag(Sigma[:r])  # TRUNCATE SIGMA TO FIRST r VALUES
        VT = VT[:r, :]  # TRUNCATE VT TO FIRST r ROWS
    A_tilde = U.T.conj() @ X2 @ VT.T.conj() @ np.linalg.inv(Sigma)  # CALCULATE APPROXIMATE LINEAR OPERATOR
    eigenvalues, W = np.linalg.eig(A_tilde)  # COMPUTE EIGENVALUES AND EIGENVECTORS
    Phi = X2 @ VT.T.conj() @ np.linalg.inv(Sigma) @ W  # PROJECT EIGENVECTORS BACK TO ORIGINAL SPACE
    return eigenvalues, Phi  # RETURN EIGENVALUES AND DYNAMIC MODES

# DEFINE ENCODER MODULE USING PYTORCH
class Encoder(nn.Module):  # ENCODER CLASS
    def __init__(self, input_dim, latent_dim):  # INITIALIZE ENCODER
        super(Encoder, self).__init__()  # CALL SUPERCLASS INIT
        self.linear1 = nn.Linear(input_dim, 128)  # LINEAR LAYER TO 128 UNITS
        self.relu = nn.ReLU()  # RELU ACTIVATION FUNCTION
        self.linear2 = nn.Linear(128, latent_dim)  # LINEAR LAYER TO LATENT DIMENSION

    def forward(self, x):  # FORWARD PASS FUNCTION
        x = self.linear1(x)  # APPLY FIRST LINEAR LAYER
        x = self.relu(x)  # APPLY RELU ACTIVATION
        x = self.linear2(x)  # APPLY SECOND LINEAR LAYER
        return x  # RETURN ENCODED OUTPUT

# DEFINE DECODER MODULE USING PYTORCH
class Decoder(nn.Module):  # DECODER CLASS
    def __init__(self, latent_dim, output_dim):  # INITIALIZE DECODER
        super(Decoder, self).__init__()  # CALL SUPERCLASS INIT
        self.linear1 = nn.Linear(latent_dim, 128)  # LINEAR LAYER TO 128 UNITS
        self.relu = nn.ReLU()  # RELU ACTIVATION FUNCTION
        self.linear2 = nn.Linear(128, output_dim)  # LINEAR LAYER TO OUTPUT DIMENSION

    def forward(self, x):  # FORWARD PASS FUNCTION
        x = self.linear1(x)  # APPLY FIRST LINEAR LAYER
        x = self.relu(x)  # APPLY RELU ACTIVATION
        x = self.linear2(x)  # APPLY SECOND LINEAR LAYER
        return x  # RETURN DECODED OUTPUT

# DEFINE AUTOENCODER MODULE
class Autoencoder(nn.Module):  # AUTOENCODER CLASS
    def __init__(self, input_dim, latent_dim):  # INITIALIZE AUTOENCODER
        super(Autoencoder, self).__init__()  # CALL SUPERCLASS INIT
        self.encoder = Encoder(input_dim, latent_dim)  # INITIALIZE ENCODER
        self.decoder = Decoder(latent_dim, input_dim)  # INITIALIZE DECODER

    def forward(self, x):  # FORWARD PASS FUNCTION
        z = self.encoder(x)  # ENCODE INPUT
        x_recon = self.decoder(z)  # DECODE LATENT REPRESENTATION
        return x_recon  # RETURN RECONSTRUCTED OUTPUT

# FUNCTION TO TRAIN THE AUTOENCODER
def train_autoencoder(model, data, epochs=100, learning_rate=1e-3):  # TRAINING FUNCTION
    criterion = nn.MSELoss()  # MEAN SQUARED ERROR LOSS
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # ADAM OPTIMIZER
    for epoch in range(epochs):  # LOOP OVER EPOCHS
        optimizer.zero_grad()  # RESET GRADIENTS
        outputs = model(data)  # FORWARD PASS
        loss = criterion(outputs, data)  # COMPUTE LOSS
        loss.backward()  # BACKPROPAGATE
        optimizer.step()  # UPDATE PARAMETERS
        if (epoch + 1) % 10 == 0:  # EVERY 10 EPOCHS
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')  # PRINT LOSS

# MAIN EXECUTION
if __name__ == "__main__":  # ENTRY POINT
    emotion_data = load_images('task-emotion')  # LOAD IMAGES FROM task-emotion FOLDER
    rest_data = load_images('task-rest')  # LOAD IMAGES FROM task-rest FOLDER
    X = np.concatenate((emotion_data, rest_data), axis=0)  # COMBINE DATASETS
    X1 = X[:-1].T  # CREATE X1 FROM DATASET
    X2 = X[1:].T  # CREATE X2 FROM DATASET

    eigenvalues, Phi = dmd(X1, X2, r=10)  # RUN DMD
    print("Eigenvalues:", eigenvalues)  # PRINT EIGENVALUES

    input_dim = X1.shape[0]  # SET INPUT DIMENSION
    latent_dim = 10  # SET LATENT DIMENSION
    model = Autoencoder(input_dim, latent_dim)  # INITIALIZE AUTOENCODER

    data_tensor = torch.tensor(X1.T, dtype=torch.float32)  # CONVERT DATA TO TENSOR
    train_autoencoder(model, data_tensor)  # TRAIN AUTOENCODER

    with torch.no_grad():  # DISABLE GRADIENT COMPUTATION
        encoded_data = model.encoder(data_tensor).numpy()  # ENCODE DATA

    X1_latent = encoded_data[:-1].T  # CREATE X1 IN LATENT SPACE
    X2_latent = encoded_data[1:].T  # CREATE X2 IN LATENT SPACE
    eigenvalues_latent, Phi_latent = dmd(X1_latent, X2_latent, r=latent_dim)  # RUN DMD IN LATENT SPACE
    print("Latent space eigenvalues:", eigenvalues_latent)  # PRINT LATENT SPACE EIGENVALUES

