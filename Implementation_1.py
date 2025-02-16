#DEEP LEARNING ENHANCED DMD (Alford-Lago et al., 2022)
import numpy as np
from pydmd import DMD
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# SYNTH DATA
def f(z, c, A):
    """ Dynamical system iteration function """
    result = (A @ z) ** 2 + c
    result[np.abs(result) > 1e10] = 1e10  # Prevent overflow
    return result

def generate_data(A, c_range, num_points, num_iter, z_dim):
    """ Generates trajectory data for different values of c """
    c_values = np.random.uniform(c_range[0], c_range[1], num_points) + \
               1j * np.random.uniform(c_range[0], c_range[1], num_points)
    data = []
    for c in c_values:
        z = np.zeros((z_dim,), dtype=np.complex128)
        trajectory = []
        for _ in range(num_iter):
            z = f(z, c, A)
            trajectory.append(z)
        data.append(np.array(trajectory))
    return np.array(data), c_values

# DATA GEN
z_dim = 2
A = np.random.uniform(-2, 2, (z_dim, z_dim)) + 1j * np.random.uniform(-2, 2, (z_dim, z_dim))
c_range = (-2, 2)
num_points = 100
num_iter = 200

data, c_values = generate_data(A, c_range, num_points, num_iter, z_dim)
data /= np.max(np.abs(data))
data_real_imag = np.stack([data.real, data.imag], axis=-1).reshape(num_points, num_iter, -1)

# DEEP AUTOENCODER BUILDING
def build_deep_dmd_autoencoder(input_shape):
    """ Deep autoencoder architecture for enhanced DMD """
    input_layer = Input(shape=input_shape)
    flat = Flatten()(input_layer)
    encoded = Dense(64, activation='relu')(flat)
    encoded = Dense(32, activation='relu')(encoded)
    latent = Dense(16, activation='relu')(encoded)  # Deeper latent space

    decoded = Dense(32, activation='relu')(latent)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(np.prod(input_shape), activation='tanh')(decoded)
    decoded = Reshape(input_shape)(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# DEEP AUTOENCODER TRAINING
input_shape = data_real_imag.shape[1:]
deep_autoencoder = build_deep_dmd_autoencoder(input_shape)
deep_autoencoder.fit(data_real_imag, data_real_imag, epochs=50, batch_size=16, verbose=1)

# ====== APPLY ZE DMD ======
def apply_dmd(data):
    """ Applies Dynamic Mode Decomposition (DMD) to the data """
    dmd = DMD()
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    data += np.random.normal(0, 1e-8, data.shape)
    dmd.fit(data.T)
    return dmd

# DMD MODES EXTRACTION
dmd = apply_dmd(data[0].real.T)

# SEE
plt.figure()
for mode in dmd.modes.T:
    plt.plot(mode.real, mode.imag, 'o-', label='DMD Mode')
plt.title('DMD Modes - Deep Learning Enhanced')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend()
plt.show()
