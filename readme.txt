@Autor: Clemens Müller

Code repository for the work on my masters project at NTNU: "TITLE", supervised by: Olav B. Fosso

Goal: Implement a physics informed neural network to solve power flows in distribution grids.

The "main" file in this code repository is the NN_obj_based. Important hyperparameters are set upon a call of
 NN_obj_based. This function trains, evaluates and stores checkpoints for each epoch.

The stored data can be visualized using plot_performance.py.

The script  essentially iterates through a for loop testing neural networks trained using different loss functions.
During each iteration a NN object is trained, and checkpoints are generated for each epoch.
Once trained, the checkpoints are loaded into separate NN instances for evaluation. This is carried out using its class methods.

Note that the custom loss functions depend on the Y-bus matrix generated using the module LF_3bus.

Datasets used for training have been generated using both the LF_3bus and PyDSAL module.

All checkpoints for neural networks trained during the work on this thesis are available through thesis supervisor.