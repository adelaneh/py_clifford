# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from math import exp
import sys


def get_orientation_tuned_firing_rate_response(stimulus, 
                                               resolution = 32,
                                               sharpness = 4.0,
                                               max_spike = 50,
                                               noise_value = .5,
                                              ):
    """
    Return the firing rate of V1 orientation-tuned cells in response to an input stimulus.
    firing rates = max_spike * ( 0.5 * ( cos(2 * (preferred_orientations - stimulus)) + 1 ) ).^sharpness

    Args:
        stimulus : float, scalar, orientation angle of stimulus
        resolution : int, scalar, number of V1 cells with preferred orientations uniformly distributed from 0 to 180 degree
        sharpness : float, scalar, controls the width of tuning curves; larger value for sharper tuning
        max_spike: float, scalar, for mean max spike number when pref = stim
        noise_value: float, scalar, determining variance = NoiseVal * mean, when generating noise

    Returns:
        float: tensor, stimulated firing rate of V1 orientation-tuned cells
    """
    if stimulus is None:
        raise TypeError("No stimulus provided.")
    if resolution is None:
        raise TypeError("No resolution provided.")
    if resolution <= 0:
        raise ValueError("Invalid resolution (%d)."%resolution)
    if sharpness is None:
        raise TypeError("No sharpness provided.")
    if sharpness <= 0:
        raise ValueError("Invalid sharpness (%f)."%sharpness)
    if max_spike is None:
        raise TypeError("No maximum spike count provided.")
    if max_spike <= 0:
        raise ValueError("Invalid maximum spike count (%f)."%max_spike)
    if noise_value is None:
        raise TypeError("No noise magnitude provided.")
    if noise_value <= 0:
        raise ValueError("Invalid noise magnitude (%f)."%noise_value)


    firing_rates           = np.zeros((resolution, 1), dtype=np.float32)
    preferred_orientations = np.pi * np.linspace(0, 1, num=resolution)
    stimuli                = np.ones(resolution) * stimulus
    response_means         = max_spike * ( 0.5 * ( np.cos(2 * (preferred_orientations - stimuli)) + 1) )**sharpness
    response_stds          = np.sqrt(noise_value * response_means)
    firing_rates           = np.random.normal(response_means, response_stds)
    firing_rates[np.where(firing_rates < 0)] = 0

    return firing_rates

def fill_input_at_instant(X, jj, 
                          s1_angle, s1_start, s1_end, 
                          s2_angle, s2_start, s2_end, 
                          go_cue,
                          timesteps,
                          num_inputs, noise_value, 
                         ):
    X[jj,s1_start:s1_end,:num_inputs-1] = np.tile(get_orientation_tuned_firing_rate_response(s1_angle, 
                                                                                             resolution = num_inputs - 1,
                                                                                             noise_value = noise_value, 
                                                                                            ),
                                                  (s1_end - s1_start, 1)
                                                 )
    X[jj,s2_start:s2_end,:num_inputs-1] = np.tile(get_orientation_tuned_firing_rate_response(s2_angle, 
                                                                                             resolution = num_inputs - 1, 
                                                                                             noise_value =  noise_value,
                                                                                            ),
                                                  (s2_end - s2_start,1)
                                                 )
    X[jj,go_cue:,num_inputs-1:num_inputs] = np.tile([1,],(timesteps - go_cue,1))

    return X

def generate_stimuli_cue_intervals(timesteps):
    """Generate stimulus/go cue intervals of varying starting points and lengths

    Args:
        timesteps: int, scalar, total number of time steps for the output

    Returns:
        tuple:

            - s1_start: start of first stimulus
            - s1_end(int): end of first stimulus
            - s2_start(int): start of second stimulus
            - s2_end(int): end of second stimulus
            - go_cue(int): start of go cue
    """
    if timesteps is None:
        raise TypeError("No timesteps provided.")
    if timesteps <= 0:
        raise ValueError("Invalid timesteps (%d)."%timesteps)
    if timesteps < 100:
        raise ValueError("Too few time steps.")

    s1_start = 50 # np.random.randint(10, 60)
    s1_end   = np.random.randint(s1_start + 10, s1_start + 60)
    s2_start = np.random.randint(s1_end + 10, s1_end + 60)
    s2_end   = np.random.randint(s2_start + 10, s2_start + 60)
    go_cue   = np.random.randint(s2_start, timesteps - 100) # np.random.randint(s2_end + 10, timesteps - 100)
    
    return (s1_start, s1_end, s2_start, s2_end, go_cue)

def generate_noise_for_intervals(batch_size, 
                                 timesteps, 
                                 num_units, 
                                 intervals,
                                 noise_magnitude=0.,
                                ):
    """Generate noise for various intervals

    Args:
        batch_size: int, scalar
        timesteps: int, scalar
        num_units: int, scalar
        intervals: list, intervals of the generated sequence to include noise
        noise_magnitude: float, scalar, standard deviation of the generated normal random noise.

    Returns:
        A (batch_size, timesteps, num_units)-dimensional array of noise at the specified intervals
    """
    res = np.zeros((batch_size, timesteps, num_units),dtype=np.float32)
    
    if noise_magnitude == 0. or len(intervals) == 0:
        return res

    for jj in range(batch_size):
        for bgx in range(len(intervals)):
            startx, endx = intervals[bgx][0], intervals[bgx][1]
            cur_noise_mat = np.random.normal(
                                             size=[endx - startx + 1, num_units],
                                             loc = 0.0,  
                                             scale=noise_magnitude,
                                            )
            res[jj, startx:endx+1, :] = cur_noise_mat

    return res

def is_countercw(a,b):
    """Determines how to decide whether orientation b is counterclockwise to orientation a 
    """
    return a <= b

def is_cw(a,b):
    """Determines how to decide whether orientation b is clockwise to orientation a
    """
    return a > b

def set_s2(s1_angle, delta, ccw_probability=0.5):
    """Set the second orientation delta degrees from the first orientation. 

    Args:
        s1_angle: float, scalar, the first orientation 
        delta: float, scalar, the absolute difference between the first and second orientations 
        ccw_probability: float, scalar the probability of the second orientation being counterclockwise 
            to the first orientation, or equivalently, one minus the probability of the second orientation 
            being clockwise to the second orientation

    Returns:
        The second orientation. This is always a positive number.
    """
    if np.random.rand() > ccw_probability or s1_angle - delta < 0:
        return s1_angle + delta
    else:
        return s1_angle - delta

def generate_random_orientation_pair(angular_diff_deg, rnd_prob, ):
    """Generate a pair of random orientations. With probability rnd_prob, these two orientations 
    will be chosen independently randomly between 0 and pi, and with probability (1 - rnd_prob) 
    they will be delta degrees apart where delta is uniformly randomly chosen from an interval 
    determined by angular_diff_deg

    Args:
        angular_diff_deg: float, scalar, determines how far apart the two orientations should be 
        rnd_prob: floar, scalar: probability of the the two orientations being chosen independently 

    Returns:
        tuple:

            - s1_angle, s2_angle: the first and second orientations respectively
    """
    s1_angle = np.random.rand()*np.pi

    if np.random.rand() < rnd_prob:
        s2_angle = np.random.rand()*np.pi
    else:
        delta = np.random.uniform(angular_diff_deg, 1.5 * angular_diff_deg) * np.pi/180.

        s2_angle = set_s2(s1_angle, delta)

    return s1_angle, s2_angle

def generate_trials(config,
                    batch_size=1000,
                    angular_diff_deg=None,
                    random_periods=True,
                    rnd_prob=0.5,
                    rescale_input=True,
                    angle1_deg=None, 
                    angle2_deg=None, 
                   ):
    """Generate input and output timeseries for orientation discrimination trials

    Args:
        config: dict, experimental configurations
        batch_size: int, scalar, size of the generated batch of data
        angular_diff_deg: float, scalar, determines how far apart the two orientations should be 
        rnd_prob: floar, scalar: probability of the the two orientations being chosen independently 
        angular_diff_deg: float, scalar, approximately determines how far apart the two orientations 
        rescale_input: bool, whether to rescale the points in the input timeseries
        
    Returns:
        tuple:
            - _X, _Y: float, tensor, a batch of corresponding input and output timeseries 
            - _s1s, _s2s: float, array, first and second directions corresponding to each generated input/output timeseries
    """

    if config is None:
        raise TypeError("No configuration provided.")

    _network_config             = config['network_params']
    _input_config               = _network_config['input_params']
    _num_inputs                 = _input_config['num_orituned_input_units']
    _num_inputs                += 1 if _input_config['has_go_cue_unit'] else 0

    _data_config                = config['data_params']
    _timesteps                  = _data_config['timesteps']
    _hidden_units_noise_std     = _data_config['hidden_units_noise_std']

    _output_config              = _network_config['output_params']
    _num_sincos_output_units    = _output_config['num_sincos_output_units']
    _num_ordinal_output_units   = _output_config['num_ordinal_output_units']
    _num_outputs                = _num_sincos_output_units + _num_ordinal_output_units

    _X                          = np.zeros((batch_size, _timesteps, _num_inputs),  dtype=np.float32)
    _Y                          = np.zeros((batch_size, _timesteps, _num_outputs), dtype=np.float32)
    _s1s                        = np.zeros((batch_size,),                          dtype=np.float32)
    _s2s                        = np.zeros((batch_size,),                          dtype=np.float32)

    for jj in range(batch_size):
        if random_periods:
            s1_start, s1_end, s2_start, s2_end, go_cue = generate_stimuli_cue_intervals(_timesteps)
        else:
            s1_start, s1_end, s2_start, s2_end, go_cue = [_data_config[varname] for varname in ['s1_start', 
                                                                                                's1_end', 
                                                                                                's2_start', 
                                                                                                's2_end', 
                                                                                                'go_cue'
                                                                                               ]
                                                         ]
        if angle1_deg is None and angle2_deg is None:
            s1_angle, s2_angle = generate_random_orientation_pair(angular_diff_deg, rnd_prob, )
        elif angle1_deg is not None and angle2_deg is not None:
            s1_angle, s2_angle = angle1_deg / 180. * np.pi, angle2_deg / 180. * np.pi
        else:
            raise(RuntimeError("Logic for only one None input orientation is not implemented."))

        _s1s[jj], _s2s[jj] = s1_angle, s2_angle

        nxt_out = 0
        if _num_sincos_output_units > 0:
            _Y[jj, go_cue:_timesteps, :_num_sincos_output_units] = np.tile([np.sin(2 * s1_angle), 
                                                                            np.cos(2 * s1_angle),
                                                                            np.sin(2 * s2_angle), 
                                                                            np.cos(2 * s2_angle),
                                                                           ], (_timesteps - go_cue, 1))
            nxt_out += _num_sincos_output_units
        if _num_ordinal_output_units > 0:
            decisions = [1 if is_countercw(s1_angle, s2_angle) else 0, 
                         1 if is_cw(s1_angle, s2_angle) else 0] if _num_ordinal_output_units == 2 else [1 if is_countercw(s1_angle, s2_angle) else -1]
            _Y[jj, go_cue:_timesteps, nxt_out:nxt_out + _num_ordinal_output_units] = np.tile(decisions, (_timesteps - go_cue, 1))
                
        fill_input_at_instant(_X, jj, 
                              s1_angle, s1_start, s1_end, 
                              s2_angle, s2_start, s2_end, 
                              go_cue, 
                              _timesteps, 
                              _num_inputs, _hidden_units_noise_std,
                             )

    if rescale_input:
        _X[:, :, :_num_inputs - 1] = _X[:, :, :_num_inputs - 1] / np.max(_X[:, :, :_num_inputs - 1]) # rescale input into range (0,1)
    
    return _X, _Y, _s1s, _s2s

