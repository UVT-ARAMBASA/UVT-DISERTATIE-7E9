import numpy as np
from pydmd import DMD
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

def f(z, c, A):  # SIS DIN
    result = (A @ z) ** 2 + c
    result[np.abs(result) > 1e10] = 1e10  # Prevent overflow
    return result

def generate_data(A, c_range, num_points, num_iter, z_dim):
    c_values = np.random.uniform(c_range[0], c_range[1], num_points) + \
               1j * np.random.uniform(c_range[0], c_range[1], num_points)
    data = []
    for c in c_values:
        z = np.zeros((z_dim,), dtype=np.complex128)  # Ensure z is a complex vector
        trajectory = []
        for _ in range(num_iter):
            z = f(z, c, A)
            trajectory.append(z)
        data.append(np.array(trajectory))
    return np.array(data), c_values

z_dim = 2
A = np.random.uniform(-2, 2, (z_dim, z_dim)) + 1j * np.random.uniform(-2, 2, (z_dim, z_dim))
c_range = (-2, 2)
num_points = 100
num_iter = 200

data, c_values = generate_data(A, c_range, num_points, num_iter, z_dim)
data /= np.max(np.abs(data))
data_real_imag = np.stack([data.real, data.imag], axis=-1).reshape(num_points, num_iter, -1)

def build_autoencoder(input_shape):
    from tensorflow.keras.regularizers import l2
    input_layer = Input(shape=input_shape)
    flat = Flatten()(input_layer)
    encoded = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(flat)
    encoded = Dense(16, activation='relu', kernel_regularizer=l2(1e-4))(encoded)
    decoded = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(encoded)
    decoded = Dense(np.prod(input_shape), activation='tanh')(decoded)
    decoded = Reshape(input_shape)(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

input_shape = data_real_imag.shape[1:]
autoencoder = build_autoencoder(input_shape)
autoencoder.fit(data_real_imag, data_real_imag, epochs=250, batch_size=16, verbose=1)

def apply_dmd(data):
    dmd = DMD()
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    data += np.random.normal(0, 1e-8, data.shape)
    dmd.fit(data.T)
    return dmd

dmd = apply_dmd(data[0].real.T)

plt.figure()
for mode in dmd.modes.T:
    plt.plot(mode.real, mode.imag, 'o-', label='DMD Mode')
plt.title('DMD Modes')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend()
plt.show()

def print_normalised():
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

print_normalised()

def visualize_mandelbrot(autoencoder, c_range, z_dim):
    resolution = 100
    c_grid = np.linspace(c_range[0], c_range[1], resolution) + \
             1j * np.linspace(c_range[0], c_range[1], resolution)[:, None]
    mandelbrot_set = np.zeros_like(c_grid, dtype=bool)
    for i in range(c_grid.shape[0]):
        for j in range(c_grid.shape[1]):
            c = c_grid[i, j]
            z = np.zeros((1, z_dim, 2))
            for _ in range(20):
                z = autoencoder.predict(z)
                if np.linalg.norm(z) > 2:
                    break
            else:
                mandelbrot_set[i, j] = True
    plt.figure()
    plt.imshow(mandelbrot_set, extent=(c_range[0], c_range[1], c_range[0], c_range[1]), cmap='hot')
    plt.title('Approximated Mandelbrot Set')
    plt.colorbar()
    plt.show()

visualize_mandelbrot(autoencoder, c_range, z_dim)
