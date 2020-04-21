==============
Simulation API
==============

--------------------
Firing Rate RNN Cell
--------------------
.. autoclass:: py_clifford.FiringRateRNNCell

-----------------------------------------------------
Firing Rate RNN for 2-line Visual Discrimination Task
-----------------------------------------------------
.. autoclass:: py_clifford.VisualDiscriminationFRRNN
   :members: train, test, save, load, __step__, __del__

-------------------------
Data Generation Utilities
-------------------------
.. automodule:: py_clifford.data_generators
   :members: get_orientation_tuned_firing_rate_response, generate_stimuli_cue_intervals, generate_noise_for_intervals, generate_random_orientation_pair, generate_trials

