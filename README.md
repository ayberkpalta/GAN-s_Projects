## GANs_Projects

This project aims to generate handwritten digits from the MNIST dataset using a basic GAN (Generative Adversarial Network) architecture using PyTorch.

Two basic neural network models are defined within the scope of the project:

Generator: Generates realistic handwritten digits from a random noise vector.

Discriminator: Tries to distinguish between real MNIST digits and fake digits.

The project shows the step-by-step training process of the GAN and provides visualization of real and fake images with TensorBoard during training.

Technical Details:
Libraries: PyTorch, torchvision, TensorBoard

Dataset: MNIST handwritten digits

Model Architecture:

Discriminator: 784 inputs, 128 hidden neurons, output 1 (real/fake)

Generator: 64-dimensional noise input, 256 hidden neurons, output 784 (image)

Activation Functions: LeakyReLU, Sigmoid, Tanh

Optimization: Adam

Loss Function: Binary Cross Entropy (BCE)

Visualization with TensorBoard: Tracking fake and real images during training
