from abc import ABC

import tensorflow as tf
from keras.engine import data_adapter

#class CustomSequential(tf.keras.Sequential):

class CustomModel(tf.keras.Sequential):

    '''
    Started developing this model in an attempt to enable graph execution with the custom loss functions.
    '''

    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happens in fit](
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.
        This method should contain the mathematical logic for one step of
        training.  This typically includes the forward pass, loss calculation,
        backpropagation, and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function`
        and `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Args:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            print(y_pred)
            layer_3_output = self.layers[3].output
            #output = graph.get_tensor_by_name('output')
            '''
            sess = tf.compat.v1.Session()
            sess.run(tf.compat.v1.global_variables_initializer())
            np_array = y_pred.eval(session=sess)
            '''
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)
