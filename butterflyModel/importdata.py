"""Converts data from MATLAB .mat files into usable python data.

This function extracts data in MATLAB's .mat file format and converts it
into usable python data.

This module will run as a script and will not be called or referenced
elsewhere in this project

Imports and collects data into a ButterflyData object, to be saved

"""
import scipy.io


def convert_mat_matrix(filename, matname):
    """Convert a MATLAB .mat file(filename) containing a matrix(matname)
    into a numpy.ndarray

    filename = name of .mat file with path, e.g. 'matrixfiles/mymatrix.mat'
    matname = name of matrix saved in .mat file, e.g. 'myMatrix'

    """
    mat_file = scipy.io.loadmat(filename)
    matrix = mat_file[matname]
    return matrix


class ButterflyData(object):
    def __init__(self):


wing_array = convert_mat_matrix('MATLAB_files/TreeNymphLeftWing.mat', 'Wing')
