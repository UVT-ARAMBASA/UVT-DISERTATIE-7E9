# Mandelbrot-DMD: Learning Fractal Structures with Autoencoders and Dynamic Mode Decomposition

This project explores **nonlinear dynamical systems** using **Dynamic Mode Decomposition (DMD) and Autoencoders**. It approximates the **Mandelbrot set** by learning the evolution of complex iterative systems.

## 📦 Installation

To set up the project, install the required dependencies using:

```sh
pip install -r requirements.txt
```

This command will install for you all the required packages which are themselves contained in the `requirements.txt` file.

### Datasets

The dataset consists of input-output pairs. These pairs represent the solutions of the nonlinear dynamical system and their corresponding classification within the Mandelbrot set. The input data consists of the iterative solutions of the system defined as:
```math
z_{n+1} = (A z_n)^2 + c
```
where:
- $z_n \in \mathbb{C}^d$ (a complex vector in $d$ dimensions),
- $A \in \mathbb{R}^{d \times d}$ (a real-valued matrix),
- $c \in \mathbb{C}$ (a complex parameter).

The system evolves over a predefined number of iterations while tracking the behavior of $z_n$ for various sampled values of $c$.

## Mathematical Formulation

This iterative system generalizes the well-known Mandelbrot and Julia sets to higher dimensions using a transformation matrix $A$. The evolution of $z_n$ depends on:
- The choice of $A$ (which may introduce stretching, rotation, or other transformations).
- The complex parameter $c$.
- The initial condition $z_0$.

### Research and Methods

This project builds on state-of-the-art methods:

- DMD for spectral analysis of complex systems
- Autoencoders for latent space compression
- Parametric and recurrent models for learning long-term behavior

The models are based on works such as:

- Alford-Lago et al. (2022), *Deep learning enhanced dynamic mode decomposition*,
  Chaos, vol. 32 ([doi:10.1063/5.0073893](https://doi.org/10.1063/5.0073893)).
- Korda & Mezić (2018), *Linear Predictors for Nonlinear Dynamical Systems: Koopman Operator Meets Model Predictive Control*,
  Automatica, vol. 93 ([doi:10.1016/j.automatica.2018.03.046](https://doi.org/10.1016/j.automatica.2018.03.046)).
- Otto & Rowley (2019), *Linearly Recurrent Autoencoder Networks for Learning Dynamics*,
  SIAM Journal on Applied Dynamical Systems, vol. 18, ([doi:10.1137/18m1177846](https://doi.org/10.1137/18m1177846)).

### Future Work

1. More structured workflow
2. Improved Visualisation
3. Pretrained models for faster testing if necessary

