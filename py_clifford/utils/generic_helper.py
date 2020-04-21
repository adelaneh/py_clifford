# coding=utf-8

import os
import numpy as np

from py_clifford.utils import install_path

def get_install_path():
    path_list = install_path.split(os.sep)
    return os.sep.join(path_list[0:len(path_list) - 1])

def get_theta_from_sin_cos_2x(sin_2x_, cos_2x_):
    if sin_2x_ is None or cos_2x_ is None:
        raise TypeError("None sin/cos value not permitted.")

    if sin_2x_ >  1.: sin_2x_ =  1.
    if sin_2x_ < -1.: sin_2x_ = -1.
    if cos_2x_ >  1.: cos_2x_ =  1.
    if cos_2x_ < -1.: cos_2x_ = -1.
        
    r1 = np.arcsin(sin_2x_)
    r2 = np.arccos(cos_2x_)
    res = None
    
    if   r1 >= 0 and r2 <= np.pi/2.: res = (r1 + r2) / 2.
    elif r1 >= 0 and r2 >  np.pi/2.: res = (np.pi - r1 + r2) / 2.
    elif r1 <  0 and r2 >  np.pi/2.: res = (np.pi - r1 + 2*np.pi - r2) / 2.
    elif r1 <  0 and r2 <= np.pi/2.: res = (2*np.pi + r1 + 2*np.pi - r2) / 2.
    
    if res is not None:
        res = (res % (2.*np.pi)) / 2.
    else:
        print("%s / %s could not be resolved."%(str((sin_2x_, cos_2x_)), str((r1, r2))))
        res = np.nan

    return res


