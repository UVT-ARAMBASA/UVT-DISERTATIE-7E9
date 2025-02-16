#Parametric DMD with Koopman MPC (Korda & Mezić, 2018)
import numpy as np
from pydmd import DMD
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# SYNTH DATA
def f(z, c, A):
    """ Dynamical system iteration function with parametric dependency """
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

# DATA GEN
z_dim = 2
A = np.random.uniform(-2, 2, (z_dim, z_dim)) + 1j * np.random.uniform(-2, 2, (z_dim, z_dim))
c_range = (-2, 2)
num_points = 50
num_iter = 200

data, c_values = generate_data(A, c_range, num_points, num_iter, z_dim)
data /= np.max(np.abs(data))
data_real_imag = np.stack([data.real, data.imag], axis=-1).reshape(num_points, num_iter, -1)

# ====== Build Parametric DMD Autoencoder ======
def build_parametric_dmd_autoencoder(input_shape):
    """ Autoencoder with parametric Koopman-inspired linear layer """
    input_layer = Input(shape=input_shape)
    flat = Flatten()(input_layer)
    encoded = Dense(64, activation='relu')(flat)
    encoded = Dense(32, activation='relu')(encoded)
    latent = Dense(16, activation='linear')(encoded)  # Linear constraint for Koopman invariant subspaces

    decoded = Dense(32, activation='relu')(latent)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(np.prod(input_shape), activation='tanh')(decoded)
    decoded = Reshape(input_shape)(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# PARAM AUTOENCODER TRAIN
input_shape = data_real_imag.shape[1:]
parametric_autoencoder = build_parametric_dmd_autoencoder(input_shape)
parametric_autoencoder.fit(data_real_imag, data_real_imag, epochs=50, batch_size=16, verbose=1)

# ====== Parametric DMD using Ridge Regression ======
class ParametricDMD:
    def __init__(self, param_order=2, alpha=1e-3):
        """ Parametric DMD with regression-based mode estimation """
        self.param_order = param_order
        self.alpha = alpha
        self.A_matrices = []
        self.param_values = []
        self.regressor = None

    def fit(self, X_list, X_prime_list, params):
        """ Fit parametric DMD to datasets across parameter values """
        assert len(X_list) == len(X_prime_list) == len(params), "Dataset size mismatch."

        A_list = []
        for X, X_prime in zip(X_list, X_prime_list):
            U, S, Vh = np.linalg.svd(X, full_matrices=False)
            S_inv = np.diag(1 / S)
            A_tilde = X_prime @ Vh.T @ S_inv @ U.T
            A_list.append(A_tilde)

        self.A_matrices = np.array(A_list)
        self.param_values = np.array(params)

        # Fit polynomial regression model
        self._fit_regression()

    def _fit_regression(self):
        """ Perform regression to learn the relationship between A matrices and parameters """
        param_matrix = np.vander(self.param_values, N=self.param_order + 1, increasing=True)
        self.regressor = Ridge(alpha=self.alpha, fit_intercept=False)
        A_flat = self.A_matrices.reshape(len(self.param_values), -1)
        self.regressor.fit(param_matrix, A_flat)

    def predict(self, new_param):
        """ Predict A matrix for a new parameter value """
        param_vector = np.vander([new_param], N=self.param_order + 1, increasing=True).reshape(1, -1)
        A_pred_flat = self.regressor.predict(param_vector).reshape(self.A_matrices.shape[1:])
        return A_pred_flat

# PARAM DMD TRAIN
pDMD = ParametricDMD(param_order=2, alpha=1e-3)
pDMD.fit([data[i, :, 0] for i in range(num_points)],
         [data[i, :, 0] for i in range(1, num_points)],
         np.linspace(c_range[0], c_range[1], num_points))

# Predict a new DMD operator for an unseen parameter
new_param = 0.55
A_pred = pDMD.predict(new_param)

# ANALYZE EIGEN-VALS
eigenvalues, eigenvectors = np.linalg.eig(A_pred)

# SEE
plt.figure()
plt.scatter(eigenvalues.real, eigenvalues.imag, marker='o', color='red', label='DMD Eigenvalues')
plt.axhline(0, color='black', linestyle="--", linewidth=0.5)
plt.axvline(0, color='black', linestyle="--", linewidth=0.5)
plt.title("Parametric DMD Eigenvalues (Korda & Mezić, 2018)")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.grid(True)
plt.show()
