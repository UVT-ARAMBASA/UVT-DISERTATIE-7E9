#Linearly-Recurrent Autoencoder (Otto & Rowley, 2019)
import numpy as np
from pydmd import DMD
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# SYNTH DATA
def f(z, c, A):
    """ Dynamical system iteration function with recurrent structure """
    result = (A @ z) ** 2 + c
    result[np.abs(result) > 1e10] = 1e10  # Prevent overflow
    return result

def generate_data(A, c_range, num_points, num_iter, z_dim):
    """ Generates trajectory data for different values of c """
    c_values = np.linspace(c_range[0], c_range[1], num_points) + \
               1j * np.linspace(c_range[0], c_range[1], num_points)[:, None]
    data = []
    for i in range(c_values.shape[0]):
        for j in range(c_values.shape[1]):
            c = c_values[i, j]
            z = np.zeros((z_dim,), dtype=np.complex128)
            trajectory = []
            for _ in range(num_iter):
                z = f(z, c, A)
                trajectory.append(z)
            data.append(np.array(trajectory))
    return np.array(data), c_values

# DATA HEN
z_dim = 2
A = np.random.uniform(-2, 2, (z_dim, z_dim)) + 1j * np.random.uniform(-2, 2, (z_dim, z_dim))
c_range = (-2, 2)
num_points = 50
num_iter = 200

data, c_values = generate_data(A, c_range, num_points, num_iter, z_dim)
data /= np.max(np.abs(data))
data_real_imag = np.stack([data.real, data.imag], axis=-1).reshape(num_points, num_iter, -1)

# LINEARLY RECURRENT AUTOENCODER BUILD
def build_recurrent_autoencoder(input_shape):
    """ Builds a linearly-recurrent autoencoder based on Otto & Rowley (2019). """
    input_layer = Input(shape=input_shape)
    encoded = LSTM(32, activation='relu', return_sequences=True)(input_layer)
    encoded = LSTM(16, activation='relu', return_sequences=False)(encoded)
    latent = Dense(16, activation='linear')(encoded)  # Ensuring linearity in latent space

    decoded = RepeatVector(input_shape[0])(latent)  # Expand latent representation
    decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(input_shape[-1]))(decoded)  # Reconstruct output

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# LINEARLY RECURRENT AUTOENCODER TRAIN
input_shape = data_real_imag.shape[1:]
recurrent_autoencoder = build_recurrent_autoencoder(input_shape)
recurrent_autoencoder.fit(data_real_imag, data_real_imag, epochs=50, batch_size=16, verbose=1)

# APPLY DMD
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
plt.title('DMD Modes - Linearly-Recurrent Autoencoder')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend()
plt.show()
