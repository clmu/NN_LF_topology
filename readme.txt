About the repository:

Code repository for the work on my masters project at NTNU: "TITLE", supervised by: Olav B. Fosso

Goal: Implement a physics informed neural network to solve power flows in distribution grids.

The "main" file in this code repository is NN_function, containing the function NN_obj_based. A call of this function
sets a number of important parameters. This function trains, evaluates and stores checkpoints for each epoch.

The stored data can be visualized using plot_performance.py.

NN_obj_based essentially iterates through a for loop training neural networks using different loss functions.
During each iteration a NN object is trained, and checkpoints are generated for each epoch.
Once trained, the checkpoints are loaded into separate NN instances for evaluation. The evaluation itself is carried out
using NeuralNetwork class methods.

Note that all custom loss functions depend on the Y-bus matrix generated using the module LF_3bus.

Datasets used for training have been generated using both the LF_3bus and PyDSAL module. Please note that PyDSAL is
subject to a similar copyright notice given in DistLoafFlow_v2.py.

All checkpoints for neural networks trained during the work on this thesis are available through thesis supervisor.

And finally: Some of the training sessions in this thesis have been conducted on Idun. For a quick introduction to
Idun, see how2idun