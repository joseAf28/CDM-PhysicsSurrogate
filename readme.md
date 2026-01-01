# CONDITIONAL DENOISING MODEL AS A PHYSICAL SURROGATE MODEL

This repository contains the code to reproduce the results and architecture presented in our paper:

> ** CONDITIONAL DENOISING MODEL AS A PHYSICAL SURROGATE MODEL**

---

This work introduces the Progressive Conditional Denoising Autoencoder (PCDAE), a generative surrogate model that overcomes the training difficulties of PINNs by implicitly learning the physical solution manifold. Experiments on a plasma physics benchmark (`data/data_3000_points.txt`) show our model is more data- and parameter-efficient than a baseline PINN with a Projection method. We also find that the PCDAE's data-driven grasp of the physics is so effective that enforcing explicit conservation laws during inference offers no additional benefit.
