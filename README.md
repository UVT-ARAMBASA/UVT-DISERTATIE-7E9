eural networks have proven their utility in solving a wide range of complex
tasks. However, their ability to approximate and analyze nonlinear dynam-
ics in real-world systems remains an open challenge. Recent approaches such
as Dynamic Mode Decomposition (DMD) have shown promise in analyzing
high-dimensional data. By combining DMD with Autoencoders, a nonlinear
dimensionality reduction technique, we propose a framework for approximat-
ing nonlinear dynamics efficiently.
DMD is a data-driven method that decomposes complex spatiotemporal
data into dynamic modes, capturing the essential features of the underlying
system dynamics. This method, particularly when extended with control,
has found applications in fields such as fluid dynamics ?, neurobiology, and
epidemiology ?. Similarly, Autoencoders excel at capturing nonlinear re-
lationships in data and can enhance the DMD approach by providing an
efficient latent space for dynamic mode extraction.
The Koopman operator theory provides a linear framework for analyzing
nonlinear dynamical systems. By applying DMD to the latent space learned
by the Autoencoder, we can extract dynamic modes that approximate the
system’s behavior over time.
