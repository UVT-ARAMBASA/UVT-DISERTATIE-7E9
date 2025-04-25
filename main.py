import numpy as np  # NUMPY FOR MATRIX AND COMPLEX OPERATIONS
from pydmd import DMD  # DYNAMIC MODE DECOMPOSITION PACKAGE
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape  # KERAS LAYERS FOR AUTOENCODER
from tensorflow.keras.models import Model  # TO DEFINE AND COMPILE MODEL STRUCTURE
import matplotlib.pyplot as plt  # PLOTTING LIBRARY FOR VISUALIZATIONS
import os  # OS FOR FILE CHECKS (WEIGHTS, ETC.)
from tensorflow.keras.activations import swish, gelu
import tensorflow as tf
import random

# BEHOLD MY SEED
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
act_func = 'gelu';
act_func_fin = 'linear';
DATA_SET = "data-set/task-rest.npz"
TRAIN_EPOCHS = 100
v_TRESH = 150 # CHANGE LATER ? (2.0 CLASSIC, FIKL-KASLIK VALUES [100,200])


def load_matrices_from_npz(file_path):
    """
    Load the 'matrices' key from a .npz file. Assumes shape (48, 200, 200).

    Args:
        file_path (str): Path to the .npz file.

    Returns:
        np.ndarray: A numpy array of shape (48, 200, 200).
    """
    data = np.load(file_path)
    if "matrices" not in data:
        raise KeyError(f"'matrices' key not found in {file_path}")
    matrices = data["matrices"]
    if matrices.shape != (48, 200, 200):
        raise ValueError(f"Expected shape (48, 200, 200), but got {matrices.shape}")
    return matrices

def f(z, c, A):  # SIS DIN
    result = np.power(A @ z, 2) + c  # HADAMARD FLOW
    result[np.abs(result) > 1e10] = 1e10  # CAP VALUES TO AVOID NUMERICAL EXPLOSION
    return result  # RETURN UPDATED STATE VECTOR

def extract_encoder(autoencoder):  # RETURNS A MODEL THAT MAPS THE INPUT TO THE ENCODED LATENT SPACE
    encoded_layer = autoencoder.get_layer(index=3).output  # GET ENCODED LAYER OUTPUT (MANUAL INDEXING, MAY VARY)
    encoder = Model(inputs=autoencoder.input, outputs=encoded_layer)  # DEFINE ENCODER MODEL STRUCTURE
    return encoder  # RETURN ENCODER MODEL

def build_classifier_from_encoder(autoencoder, classifier_output_activation='sigmoid'):
    """Returns a classifier model built on top of the encoder."""
    encoder = extract_encoder(autoencoder)  # GET ENCODER FROM AUTOENCODER
    encoded_input = encoder.output  # GET ENCODED REPRESENTATION
    classification_output = Dense(1, activation=classifier_output_activation)(encoded_input)
    #^ CLASSIFIER LAYER ON LATENT SPACE
    classifier = Model(inputs=encoder.input, outputs=classification_output)  # BUILD FULL CLASSIFIER MODEL
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #^ COMPILE FOR BINARY CLASSIFICATION
    return classifier  # RETURN COMPILED CLASSIFIER

def generate_data(A, c_range, num_points, num_iter, z_dim):  # GENERATES SYSTEM DYNAMICS FOR RANDOM c VALUES
    c_values = np.random.uniform(c_range[0], c_range[1], num_points) + \
               1j * np.random.uniform(c_range[0], c_range[1], num_points)  # UNIFORMLY SAMPLE COMPLEX c VALUES
    data = []  # STORE ALL TRAJECTORIES
    for c in c_values:  # FOR EACH RANDOM c VALUE
        z = np.zeros((z_dim,), dtype=np.complex128)  # INIT z = 0 (VECTOR), COMPLEX
        trajectory = []  # INIT TRAJECTORY STORAGE
        for _ in range(num_iter):  # ITERATE SYSTEM FOR num_iter STEPS
            z = f(z, c, A)  # APPLY DYNAMICS
            trajectory.append(z)  # STORE RESULT
        data.append(np.array(trajectory))  # ADD TRAJECTORY TO FULL DATASET
    return np.array(data), c_values  # RETURN DATA AND CORRESPONDING c VALUES

z_dim = 200  # DIMENSION OF z (e.g., ℂ²)
#A = np.random.uniform(-2, 2, (z_dim, z_dim)) + 1j * np.random.uniform(-2, 2, (z_dim, z_dim))  #C BELOW
# RANDOM COMPLEX MATRIX A
all_matrices = load_matrices_from_npz(DATA_SET)  # or "task-emotion.npz"
A = all_matrices[0]  # or iterate through them

c_range = (-0.43, 0.43)  # RANGE FOR REAL AND IMAG PARTS OF c
num_points = 100  # HOW MANY RANDOM c VALUES TO SAMPLE
num_iter = 200  # HOW MANY ITERATIONS PER TRAJECTORY
data, c_values = generate_data(A, c_range, num_points, num_iter, z_dim)  # GENERATE TRAINING DATA
#data /= np.max(np.abs(data))
#COMMENTED TO REMOVE NORMALISATION (WILL I NEED IT, IDFK)
data_real_imag = np.stack([data.real, data.imag], axis=-1).reshape(num_points, num_iter, -1)  #SEE BELOW
# FLATTEN REAL + IMAG INTO FINAL SHAPE

def build_autoencoder(input_shape):  # BUILD AUTOENCODER MODEL
    from tensorflow.keras.regularizers import l2  # L2 REGULARIZATION TO PREVENT OVERFITTING
    input_layer = Input(shape=input_shape)  # INPUT LAYER
    flat = Flatten()(input_layer)  # FLATTEN INPUT
    encoded = Dense(32, activation=act_func, kernel_regularizer=l2(1e-4))(flat)  # ENCODED LAYER 1
    encoded = Dense(16, activation=act_func, kernel_regularizer=l2(1e-4))(encoded)  # ENCODED LAYER 2
    encoded = Dense(8, activation=act_func, kernel_regularizer=l2(1e-4))(encoded)  # ENCODED LAYER 3 (LATENT SPACE)
    decoded = Dense(32, activation=act_func, kernel_regularizer=l2(1e-4))(encoded)  # DECODED LAYER 1
    decoded = Dense(np.prod(input_shape), activation=act_func_fin)(decoded)  # FINAL DENSE OUTPUT
    decoded = Reshape(input_shape)(decoded)  # RESHAPE BACK TO ORIGINAL SHAPE
    autoencoder = Model(inputs=input_layer, outputs=decoded)  # BUILD FULL AUTOENCODER MODEL
    autoencoder.compile(optimizer='adam', loss='mse')  # COMPILE WITH MSE LOSS
    return autoencoder  # RETURN AUTOENCODER

input_shape = data_real_imag.shape[1:]  # SET SHAPE FOR AUTOENCODER BASED ON DATA
autoencoder = build_autoencoder(input_shape)  # CREATE AUTOENCODER
#autoencoder.fit(data_real_imag, data_real_imag, epochs=250, batch_size=16, verbose=1)
#VALIDATION SPLIT?

autoencoder.fit(data_real_imag, data_real_imag,
                epochs=TRAIN_EPOCHS, batch_size=16, verbose=1,
                validation_split=0.1)  # TRAIN AUTOENCODER
weights_path = "autoencoder.weights.h5"  # FILE TO SAVE/LOAD WEIGHTS
# H5 GOOD, READ WHATEVER ZE GOBSHITTING-FUCK IT DOES

#SAVE THE FILE
if os.path.exists(weights_path):  # IF WEIGHTS EXIST
    autoencoder.load_weights(weights_path)  # LOAD THEM
    print("Loaded saved weights.")  # LOG LOAD
else:  # OTHERWISE
    print("No saved weights found. Training from scratch.")  # LOG TRAINING
    autoencoder.fit(data_real_imag, data_real_imag,
                    epochs=TRAIN_EPOCHS, batch_size=16, verbose=1,
                    validation_split=0.1)  # TRAIN
    autoencoder.save_weights(weights_path)  # SAVE WEIGHTS
    print("Weights saved to autoencoder.weights.h5")  # LOG SAVE

# BUILD AND TRAIN CLASSIFIER
classifier = build_classifier_from_encoder(autoencoder)  # USE ENCODER TO MAKE CLASSIFIER

# CREATE TRAINING LABELS: 1 - BOUNDED (NORM ≤ 2), OTHRWISE 0
labels = np.zeros((num_points,))  # INIT LABEL ARRAY
threshold = 2.0  # SET THRESHOLD FOR BEING IN SET
for i in range(num_points):  # FOR EACH TRAJECTORY
    if np.linalg.norm(data[i]) <= threshold:  # CHECK BOUND CONDITION
        labels[i] = 1  # MARK AS IN MANDELBROT SET

# Train classifier on same data
classifier.fit(data_real_imag, labels, epochs=30, batch_size=8, verbose=1, validation_split=0.1)  # TRAIN CLASSIFIER

def apply_dmd_OLD(data):  # OLD VERSION FOR DMD (UNUSED)
    dmd = DMD()
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    data += np.random.normal(0, 1e-8, data.shape)
    dmd.fit(data.T)
    return dmd

def apply_dmd(data):  # APPLY MODERN DMD TO REAL DATA
    dmd = DMD(svd_rank=10)  # LIMIT NUMBER OF MODES
    clean_data = np.nan_to_num(data.real.T)  # REMOVE NaNs AND TRANSPOSE
    dmd.fit(clean_data)  # FIT MODEL
    return dmd  # RETURN DMD OBJECT

for i in range(5):  # or a random sample
    dmd = apply_dmd(data[i].real.T)  # RUN DMD ON SAMPLE TRAJECTORIES
    # PLOT OR ANALYSE EACH OF 'EM

plt.figure()  # INIT PLOT
for mode in dmd.modes.T:  # FOR EACH MODE
    plt.plot(mode.real, mode.imag, 'o-', label='DMD Mode')  # PLOT AS POINTS
plt.title('DMD Modes')  # TITLE
plt.xlabel('Real Part')  # LABEL X
plt.ylabel('Imaginary Part')  # LABEL Y
plt.legend()  # ADD LEGEND
plt.show()  # DISPLAY PLOT

def print_normalised():  # PLOT TRAJECTORIES (NORMALIZED)
    plt.figure()
    for i in range(data.shape[0]):
        traj = data[i, :, 0]
        traj /= np.max(np.abs(traj))  # Normalize each trajectory
        plt.plot(traj.real, traj.imag, label=f'Trajectory {i}')
    plt.title('Normalized Trajectories of z')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.legend()
    plt.show()

def print_non_normalised():  # PLOT TRAJECTORIES (ORIGINAL SCALE)
    plt.figure()
    for i in range(data.shape[0]):
        traj = data[i, :, 0]
        plt.plot(traj.real, traj.imag, label=f'Trajectory {i}')
    plt.title('Non-Normalized Trajectories of z')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.legend()
    plt.show()

print_normalised()  # PLOT NORMALIZED TRAJECTORIES

print_non_normalised()  # PLOT RAW TRAJECTORIES

def visualize_mandelbrot_OLD(autoencoder, c_range, z_dim):  # LEGACY VERSION WITH AUTOENCODER
    resolution = 100
    c_grid = np.linspace(c_range[0], c_range[1], resolution) + \
             1j * np.linspace(c_range[0], c_range[1], resolution)[:, None]
    mandelbrot_set = np.zeros_like(c_grid, dtype=bool)
    for i in range(c_grid.shape[0]):
        for j in range(c_grid.shape[1]):
            c = c_grid[i, j]
            z = np.zeros((1, z_dim, 2))
            for _ in range(20):
                z_combined = np.stack([z.real, z.imag], axis=-1).reshape(1, z_dim, 2)
                z_pred = autoencoder.predict(z_combined)
                z = z_pred[0, :, 0] + 1j * z_pred[0, :, 1]
                if np.linalg.norm(z) > 2:
                    break
            else:
                mandelbrot_set[i, j] = True
    plt.figure()
    plt.imshow(mandelbrot_set, extent=(c_range[0], c_range[1], c_range[0], c_range[1]), cmap='hot')
    plt.title('Approximated Mandelbrot Set')
    plt.colorbar()
    plt.show()

def visualize_mandelbrot(f, A, c_range, z_dim, max_iter=20, threshold=v_TRESH):  # DIRECT DYNAMICAL SIMULATION
    resolution = 100  # NUMBER OF POINTS PER AXIS IN COMPLEX PLANE GRID
    real_vals = np.linspace(c_range[0], c_range[1], resolution)  # REAL AXIS SAMPLES
    imag_vals = np.linspace(c_range[0], c_range[1], resolution)  # IMAGINARY AXIS SAMPLES
    mandelbrot_set = np.zeros((resolution, resolution), dtype=bool)  # INITIALIZE BOOLEAN MASK FOR SET
    for i, re in enumerate(real_vals):  # LOOP OVER REAL PARTS
        for j, im in enumerate(imag_vals):  # LOOP OVER IMAG PARTS
            c = re + 1j * im  # COMBINE INTO COMPLEX PARAMETER c
            z = np.zeros((z_dim,), dtype=np.complex128)  # START TRAJECTORY FROM ORIGIN IN ℂ^z_dim
            for _ in range(max_iter):  # ITERATE MAX_ITER TIMES
                z = f(z, c, A)  # APPLY SYSTEM DYNAMICS
                if np.linalg.norm(z) > threshold:  # ESCAPE CONDITION (OUTSIDE BOUND)
                    break  # NOT IN SET → EXIT
            else:
                mandelbrot_set[j, i] = True  # IF ESCAPE DIDN’T HAPPEN, POINT IS IN SET


    plt.figure()
    plt.imshow(mandelbrot_set, extent=(c_range[0], c_range[1], c_range[0], c_range[1]),
               cmap='hot', origin='lower')
    plt.title('Approximated Mandelbrot Set (Direct Dynamics)')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.colorbar(label='In Set')
    plt.show()

visualize_mandelbrot(f, A, c_range, z_dim)  # GENERATE FRACTAL VISUALIZATION

#visualize_mandelbrot_OLD(autoencoder, c_range, z_dim)  # LEGACY AUTOENCODER VERSION


