@Autor: Clemens Müller

Code repository for the work on my masters project at NTNU: "TITLE", supervised by: Olav B. Fosso

Goal: Implement a physics informed neural network to solve power flows in distribution grids.

The "main" file in this code repository is the NN_obj_based. Network size and hyperparameters are selected just below line 83.

The script  essentially iterates through a for loop testing neural networks trained using different loss functions.
During each iteration a NN object is trained, and checkpoints are generated for each epoch.
Once trained, the checkpoints are loaded into separate NN instances for evaluation. This is carried out using its class methods.

Plotting and result analysis is carried out in a separate script loading the performance data per epoch.

Note that the custom loss functions depend on the Y-bus matrix generated using the module LF_3bus.

Datasets used for training have been generated using both the LF_3bus and PyDSAL module.

All checkpoints for neural networks trained during the work on this thesis should still be available. Please contact my supervisor should they be of interest.
