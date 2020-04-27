#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals

import os
import logging
from tqdm.auto import trange

import tensorflow as tf
from tensorflow.keras.activations import tanh, linear, relu, sigmoid
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from py_clifford.config import load_configs
from py_clifford.unit_sv_initializer import UnitSVInitializer
from py_clifford.data_generators import *
from py_clifford.firing_rate_rnn import FiringRateRNNCell
from py_clifford.timeliner import TimeLiner

class VisualDiscriminationFRRNN():
    """Firing rate recurrent neural network for 2-line visual discrimination task.
    See Ding, et. al. 2017 for more details.
    """
    def __init__(self,
                 config):
        tf.reset_default_graph()

        if config is None:
            raise TypeError("No configuration provided.")

        self._config                 = config

        self._log_level              = eval("logging."+self._config['logging_level'])
        self._logger                 = logging.getLogger(__name__)
        self._logger.setLevel(self._log_level)

        self._frrnn_config           = self._config['network_params']

        self._input_config           = self._frrnn_config['input_params']
        self._num_inputs             = int(self._input_config['num_orituned_input_units'])
        self._num_inputs            += 1 if self._input_config['has_go_cue_unit'] else 0

        self._output_config          = self._frrnn_config['output_params']
        self._num_outputs            = int(self._output_config['num_sincos_output_units']) + int(self._output_config['num_ordinal_output_units'])

        self._data_config            = self._config['data_params']
        self._data_timesteps         = int(self._data_config['timesteps'])
        self._min_ad                 = float(self._data_config['min_angular_diff'])
        self._max_ad                 = float(self._data_config['max_angular_diff'])
        self._ann_const              = float(self._data_config['sampling_annealing_const'])
        self._rnd_prob               = float(self._data_config['random_orientation_sampling_prob'])
        self._hidden_units_noise_std = float(self._data_config['hidden_units_noise_std'])

        self._X                      = tf.compat.v1.placeholder("float", [None, self._data_timesteps, self._num_inputs],  name = "X")
        self._Y                      = tf.compat.v1.placeholder("float", [None, self._data_timesteps, self._num_outputs], name = "Y")

        self._hidden_layer_config    = self._frrnn_config['hidden_params']

        self._num_hidden_units       = int(self._hidden_layer_config['num_hidden_units'])
        self._hidden_activation_func = eval(self._hidden_layer_config['activation_function'])
        self._dtovertau              = float(self._hidden_layer_config['dtovertau'])
        self._w_initializer          = eval(self._hidden_layer_config['w_initializer'])
        self._layer_normalize        = bool(self._hidden_layer_config['layer_normalize'])

        self._training_config        = self._config['training_params']
        self._training_steps         = int(self._training_config['training_steps'])
        self._num_epochs             = int(self._training_config['num_epochs'])
        self._next_ang_diff          = lambda cur_iter: self._min_ad + (self._max_ad - self._min_ad) * np.exp(-self._ann_const * (cur_iter - 1) / self._training_steps)
        self._train_batch_size       = int(self._training_config['batch_size'])

        self._persistence_params     = self._config['persistence_params']
        self._display_step           = int(self._persistence_params['display_step'])
        self._save_step              = int(self._persistence_params['save_step'])
        self._visualize_state        = lambda _s: _s % self._display_step == 0 or _s == 1 or _s == self._training_steps
        self._save_state             = lambda _s: _s % self._save_step == 0 or _s == 1 or _s == self._training_steps
        self._save_weights           = bool(self._persistence_params['save_weights'])
        self._save_test_pointcloud   = bool(self._persistence_params['save_test_pointcloud'])
        self._save_path              = self._persistence_params['save_path'] if 'save_path' in self._persistence_params else None

        self._frrnn                  = FiringRateRNNCell(
                                        self._num_hidden_units,
                                        activation=self._hidden_activation_func,
                                        dtovertau=self._dtovertau,
                                        w_initializer=self._w_initializer,
                                        layer_normalize=self._layer_normalize,
                                       )
        self._hidden_outputs, self._hidden_states = tf.nn.dynamic_rnn(self._frrnn, self._X, dtype=tf.float32)

        # Define weights
        self._weights = {
            'hidden': self._frrnn._kernel,
            'out': tf.Variable(tf.random.normal([self._num_hidden_units, self._num_outputs]), name='out_weights')
        }
        self._biases = {
            'hidden': self._frrnn._bias,
            'out': tf.Variable(tf.random.normal([self._num_outputs]), name='out_biases')
        }

        self._hidden_after_activation   = tf.reshape(self._hidden_activation_func(self._hidden_outputs, ),
                                                     [-1, self._num_hidden_units],
                                                     name = 'hidden_after_activation',
                                                    )
        self._hidden_before_acttivation = tf.reshape(self._hidden_outputs,[-1, self._num_hidden_units])

        # Linear output (no activation applied at the output layer units)
        self._linear_output = tf.matmul(self._hidden_activation_func(self._hidden_before_acttivation),
                                                                     self._weights['out']
                                                                    ) + self._biases['out']
        self._linear_output = tf.reshape(self._linear_output,[-1, self._data_timesteps, self._num_outputs], name = 'linear_output')

        self._sincos_activation_function  = eval(self._output_config['sincos_activation_function'])
        self._ordinal_activation_function = eval(self._output_config['ordinal_activation_function'])
        self._num_sincos_output_units     = int(self._output_config['num_sincos_output_units'])
        self._num_ordinal_output_units    = int(self._output_config['num_ordinal_output_units'])
        ## For linear output, set `self._hidden_layer_config['activation_function']` to `linear`
        ## To use rectified tanh for hidden layer activation, set the `self._hidden_layer_config['activation_function']` to `lambda x: relu(tanh(x))`
        self._transformed_output          = tf.concat((
                                                       self._sincos_activation_function(self._linear_output[:, :, :self._num_sincos_output_units]), 
                                                       self._ordinal_activation_function(self._linear_output[:, :, self._num_sincos_output_units:self._num_sincos_output_units+self._num_ordinal_output_units])), 
                                                       axis=2
                                                     )

        self._loss_op   = tf.reduce_mean(tf.square(self._transformed_output - self._Y),name = 'loss_op')
        self._err_op    = tf.reduce_mean(tf.square(self._transformed_output - self._Y),name = 'err_op')

        self._optimizer       = tf.compat.v1.train.AdamOptimizer(
                                           learning_rate = self._training_config['learning_rate'],
                                           beta1         = self._training_config['adam_beta1'],
                                           beta2         = self._training_config['adam_beta2'],
                                           name          = 'Adam'
                                        )
        self._gvs             = self._optimizer.compute_gradients(self._loss_op)
        self._capped_gvs      = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self._gvs]
        self._train_op        = self._optimizer.apply_gradients(self._capped_gvs)

        self._testing_config                 = self._config['testing_params']
        self._testing_batch_size             = int(self._testing_config['batch_size'])
        self._read_linear_hidden_activity    = bool(self._testing_config['read_linear_hidden_activity'])
        self._integrate_response_window_size = int(self._testing_config['integrate_response_window_size'])

        self._testing_X     = None
        self._testing_Y     = None
        self._testing_s1s   = None
        self._testing_s2s   = None

        self._validation_config             = self._config['validation_params']
        self._validation_batch_size         = int(self._validation_config['batch_size'])
        self._validation_step               = int(self._validation_config['validation_step'])
        self._validate_state                = lambda _s: _s % self._validation_step == 0 or _s == 1 or _s == self._training_steps

        self._training_loss_history               = []
        self._training_loss_history_viz           = []
        self._training_accuracy_matrix_history    = []

        self._logger.info('Network setup complete.')
        self._logger.debug('Number of sin/cos outputs=%d', self._num_sincos_output_units)
        self._logger.debug('Number of ordinal outputs=%d', self._num_ordinal_output_units)
        self._logger.debug('Learning rate=%f', self._training_config['learning_rate'])

        self._tf_params                              = self._config['tensorflow_params']
        self._glob_var_init                          = tf.compat.v1.global_variables_initializer()
        self._model_saver                            = tf.compat.v1.train.Saver()

        self._tf_config                              = tf.compat.v1.ConfigProto()
        self._tf_config.intra_op_parallelism_threads = int(self._tf_params['intra_op_parallelism_threads'])
        self._tf_config.inter_op_parallelism_threads = int(self._tf_params['inter_op_parallelism_threads'])
        self._tf_config.allow_soft_placement         = bool(self._tf_params['allow_soft_placement'])

        self._tf_session                             = tf.compat.v1.Session(config = self._tf_config)

        self._tf_session.run(self._glob_var_init)

    def __del__(self):
        """Close the tensorflow session.
        """
        self._tf_session.close()

    def __step__(self, _step=None):
        """Take a training step.

        Args:
            _step: int, step number. Used to calculate the next approximate angular difference.

        Returns:
            tuple:

                - _X, _Y: tensors, the input and output tensors used to take the step.
                - _s1s, _s2s: arrays, orientations of the trials used to take the step.
        """
        if _step is None:
            _step = 1
        self._training_step_X, self._training_step_Y, self._training_step_s1s, self._training_step_s2s = generate_trials(self._config,
                                             batch_size=self._train_batch_size,
                                             angular_diff_deg=self._next_ang_diff(_step),
                                             random_periods=True,
                                             rnd_prob=self._rnd_prob,
                                             rescale_input=True,
                                            )
        self._frrnn._is_generate_noise      = True
        with trange(self._num_epochs, desc='Epoch', leave=False) as epoch_trange:
            for epoch in epoch_trange:
                self._tf_session.run(self._train_op, feed_dict={self._X: self._training_step_X, self._Y: self._training_step_Y})

    def train(self):
        """Train the network.
        """

        with trange(1, self._training_steps+1, desc='Training step',) as training_step_trange:
            for _step in training_step_trange:
                self._logger.debug('Training step %d'%_step)

                self.__step__(_step)

                _, _training_loss = self._tf_session.run([self._transformed_output, self._loss_op], 
                                                                        feed_dict={self._X: self._training_step_X, self._Y: self._training_step_Y},
                                                                      )
                self._training_loss_history.append(_training_loss)

                training_step_trange.set_postfix(angular_diff=self._next_ang_diff(_step), training_loss=_training_loss)

                if self._validate_state(_step):
                    self._logger.debug("Validating at step %d", _step)
                    self._training_loss_history_viz.append(_training_loss)
                    self.validate(visualize=self._visualize_state(_step))
                
                if self._save_state(_step):
                    if self._save_test_pointcloud:
                        _, _testing_hidden_activity_tensor, _, _testing_s1s, _testing_s2s, _, _ = self.test(
                              testing_batch_size=self._testing_batch_size,
                        )
                        self.save(_testing_hidden_activity_tensor, _testing_s1s, _testing_s2s)
                    else:
                        self.save()

    def test(self, angle1_deg=None, angle2_deg=None, testing_batch_size=None):
        """Test the network, possibly using fixed input orientations.

        Args:
            angle1_deg: float, (optional) the fixed first input orientation
            angle2_deg: float, (optional) the fixed second input orientation
            testing_batch_size: float, (optional) the (modified) testing batch size
        """
        if testing_batch_size is None:
            testing_batch_size = self._testing_batch_size
        if self._testing_X is None or self._testing_Y is None or \
          self._testing_s1s is None or self._testing_s2s is None:
            self._testing_X, self._testing_Y, self._testing_s1s, self._testing_s2s = generate_trials(self._config,
                                                 batch_size=testing_batch_size,
                                                 angular_diff_deg=None,
                                                 random_periods=False,
                                                 rnd_prob=1.0,
                                                 rescale_input=True,
                                                 angle1_deg=angle1_deg,
                                                 angle2_deg=angle2_deg,
                                                )
        self._testing_output, self._testing_hidden_activity_tensor, self._testing_error = self._tf_session.run(
                [self._transformed_output,
                 self._hidden_before_acttivation if self._read_linear_hidden_activity else self._hidden_after_activation, 
                 self._err_op
                ], 
                feed_dict={self._X: self._testing_X, self._Y: self._testing_Y})

        return self._testing_output, self._testing_hidden_activity_tensor, self._testing_error, self._testing_s1s, self._testing_s2s, self._testing_X, self._testing_Y

    def save_model(self, _save_path):
        _save_path = self.__create_save_path__(_save_path)

        self._model_saver.save(self._tf_session, _save_path + '/model.ckpt')

    def load_model(self, _load_path):
        self._model_saver.restore(self._tf_session, _load_path + '/model.ckpt')

    def save(self, _testing_pointcloud=None, _testing_s1s=None, _testing_s2s=None, _save_path=None):
        """Save network parameters and recurrent units' activation pointcloud.

        Args:
            _testing_pointcloud: tensor, (optional) the pointcloud to be saved
            _testing_s1s: array, (optional) first orientations of the trials used to generate _testing_pointcloud
            _testing_s2s: array, (optional) second orientations of the trials used to generate _testing_pointcloud
            _save_path: string, (optionsl) path to the saving directory
        """
        if self._save_weights:
            self.save_model(_save_path)
        if self._save_test_pointcloud and \
           _testing_pointcloud is not None and \
           _testing_s1s is not None and \
           _testing_s2s is not None:
            self.__save_testing_pointcloud__(_testing_pointcloud, _testing_s1s, _testing_s2s, _save_path)

    def __create_save_path__(self, _save_path):
        if _save_path is None:
            _save_path = self._save_path
        if _save_path is None:
            _save_path = "%s_%dhidden_%doutputs_%s"%(self._hidden_layer_config['activation_function'],
                                                     self._num_hidden_units,
                                                     self._num_outputs,
                                                     str(self._hidden_units_noise_std).replace('.', '_')
                                                    )
        if not os.path.exists(_save_path):
            os.makedirs(_save_path)

        return _save_path

    def __save_testing_pointcloud__(self, _testing_pointcloud, _testing_s1s, _testing_s2s, _save_path=None):
        _save_path = self.__create_save_path__(_save_path)

        np.save(_save_path + '/ptcd', [_testing_pointcloud,])
        np.save(_save_path + '/dirs', (_testing_s1s, _testing_s2s))

    def load(self, _load_path):
        """Load network parameters.

        Args:
            _load_path: string, (optionsl) path to the directory to load the weights from
        """
        if not os.path.exists(_save_path):
            raise RuntimeError("Folder %s does not exist."%_save_path)

        self.load_model(_load_path)

    def validate(self, validation_orientations=[(50., 53.), (50., 60.), ], visualize=False):
        if validation_orientations is None:
            raise TypeError("No validation orientations provided.")
        if len(validation_orientations) == 0:
            raise ValueError("No validation orientations provided.")

        for vo_inx in range(len(validation_orientations)):
            angle1_deg, angle2_deg = validation_orientations[vo_inx]
            self._val_X_ccw, self._val_Y_ccw, self._val_s1s_ccw, self._val_s2s_ccw = generate_trials(self._config,
                                                                                 batch_size=self._validation_batch_size,
                                                                                 angular_diff_deg=None,
                                                                                 random_periods=False,
                                                                                 rescale_input=True,
                                                                                 angle1_deg=angle1_deg,
                                                                                 angle2_deg=angle2_deg,
                                                                                )
            self._val_X_cw, self._val_Y_cw, self._val_s1s_cw, self._val_s2s_cw = generate_trials(self._config,
                                                                              batch_size=self._validation_batch_size,
                                                                              angular_diff_deg=None,
                                                                              random_periods=False,
                                                                              rescale_input=True,
                                                                              angle1_deg=angle2_deg,
                                                                              angle2_deg=angle1_deg,
                                                                             )

            shuffling_indexes = np.arange(len(self._val_X_ccw) + len(self._val_X_cw))
            np.random.shuffle(shuffling_indexes)
            self._val_X = np.append(self._val_X_ccw, self._val_X_cw, axis=0)
            self._val_X = self._val_X[shuffling_indexes]
            self._val_Y = np.append(self._val_Y_ccw, self._val_Y_cw, axis=0)
            self._val_Y = self._val_Y[shuffling_indexes]

            self._val_predictions = self._tf_session.run(self._transformed_output, feed_dict={self._X: self._val_X, self._Y: self._val_Y})

            _ccw_batch_indexes     = [xx for xx in range(len(shuffling_indexes)) if shuffling_indexes[xx] <  len(self._val_X_ccw)]
            _cw_batch_indexes      = [xx for xx in range(len(shuffling_indexes)) if shuffling_indexes[xx] >= len(self._val_X_ccw)]
            _val_ccw_predictions   = self._val_predictions[_ccw_batch_indexes]
            _val_cw_predictions    = self._val_predictions[_cw_batch_indexes]
            self._val_X_ccw        = self._val_X[_ccw_batch_indexes]
            self._val_Y_ccw        = self._val_Y[_ccw_batch_indexes]
            self._val_X_cw         = self._val_X[_cw_batch_indexes]
            self._val_Y_cw         = self._val_Y[_cw_batch_indexes]

            if vo_inx == 0 and visualize:
                self._logger.debug("Visualizing (in validate) for the first pair of validation_orientations (%.02f, %.02f)", angle1_deg, angle2_deg)
                self._training_accuracy_matrix_history.append(self.__create_accuracy_matrix__(_val_ccw_predictions, _val_cw_predictions))
                self.__plot_loss_accuracy__()
                self.__visualize__(_val_ccw_predictions, _val_cw_predictions)

    def __plot_loss_accuracy__(self, _save_path=None):
        _save_path = self.__create_save_path__(_save_path)

        irws = 1
        if self._integrate_response_window_size is not None and self._integrate_response_window_size > 0:
            irws = self._integrate_response_window_size
        self._logger.debug("Attempting at plotting loss and accuracy. len(_training_loss_history_viz)=%d, irws=%d", len(self._training_loss_history_viz), irws)

        if len(self._training_loss_history_viz) > irws:
            plot_x          = np.multiply(self._display_step, range(np.size(self._training_loss_history_viz)))
            ccw_accuracies  = [tamh[2] for tamh in self._training_accuracy_matrix_history]
            cw_accuracies   = [tamh[5] for tamh in self._training_accuracy_matrix_history]

            _fig = plt.figure(figsize=(15,10))
            plt.plot(plot_x, self._training_loss_history_viz, label="Training loss")
            if self._num_ordinal_output_units > 0:
                plt.plot(plot_x, ccw_accuracies, label="Validation CCW Accuracy")
                plt.plot(plot_x, cw_accuracies, label="Validation CW Accuracy")

                plt.title("Average CW acc=%.02f, average CCW acc=%.02f (over last %d steps)"%(
                    np.mean(cw_accuracies[-irws:]),
                    np.mean(ccw_accuracies[-irws:]),
                    irws)
                )
            plt.xlabel("Steps")
            plt.ylabel("Loss/Accuracy")
            plt.ylim([0,1.0])
            plt.savefig(_save_path + "/" + "loss_acc.png", dpi=200)
            self._logger.debug("loss_acc.png is saved in %s", _save_path)
            plt.close(_fig)

        return

    def __visualize__(self, _predictions_ccw, predictions_cw):
        return

    def __create_accuracy_matrix__(self, _predictions_ccw, _predictions_cw):
        correct_ccw, incorrect_ccw, correct_cw, incorrect_cw = 0, 0, 0, 0
        irws = 1
        if self._integrate_response_window_size is not None and self._integrate_response_window_size > 0:
            irws = self._integrate_response_window_size

        if self._num_ordinal_output_units == 2:
            for jj in range(_predictions_ccw.shape[0]):
                pred_ccw_jj_1 = np.mean(_predictions_ccw[jj, self._data_timesteps-irws:self._data_timesteps, self._num_sincos_output_units])
                pred_ccw_jj_2 = np.mean(_predictions_ccw[jj, self._data_timesteps-irws:self._data_timesteps, self._num_sincos_output_units+1])
                if pred_ccw_jj_1 > pred_ccw_jj_2:
                    correct_ccw += 1
                else:
                    incorrect_ccw += 1

            for jj in range(_predictions_cw.shape[0]):
                pred_cw_jj_1  = np.mean(_predictions_cw[jj,  self._data_timesteps-irws:self._data_timesteps, self._num_sincos_output_units])
                pred_cw_jj_2  = np.mean(_predictions_cw[jj,  self._data_timesteps-irws:self._data_timesteps, self._num_sincos_output_units+1])
                if pred_cw_jj_1 < pred_cw_jj_2:
                    correct_cw += 1
                else:
                    incorrect_cw += 1
        elif self._num_ordinal_output_units == 1:
            for jj in range(_predictions_ccw.shape[0]):
                pred_ccw_jj = np.mean(_predictions_ccw[jj, self._data_timesteps-irws:self._data_timesteps, self._num_sincos_output_units])
                if pred_ccw_jj > 0:
                    correct_ccw += 1
                else:
                    incorrect_ccw += 1

            for jj in range(_predictions_cw.shape[0]):
                pred_cw_jj  = np.mean(_predictions_cw[jj, self._data_timesteps-irws:self._data_timesteps, self._num_sincos_output_units])
                if pred_cw_jj < 0:
                    correct_cw += 1
                else:
                    incorrect_cw += 1
        else:
            raise(RuntimeError('Accuracy calculation logic for %d output units is not implemented.'%self._num_ordinal_output_units))

        ccw_accuracy = 1. * correct_ccw / _predictions_ccw.shape[0]
        cw_accuracy  = 1. * correct_cw  / _predictions_cw.shape[0]

        return correct_ccw, incorrect_ccw, ccw_accuracy, correct_cw, incorrect_cw, cw_accuracy

