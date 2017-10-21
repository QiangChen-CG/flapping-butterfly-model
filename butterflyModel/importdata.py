"""Converts data from MATLAB .mat files into usable python data.

This function extracts data in MATLAB's .mat file format and converts it
into usable python data.

This module will run as a script and will not be called or referenced
elsewhere in this project

Imports and collects data into a ButterflyData object, to be saved

"""
import scipy.io
import buildbutterfly
import pickle


def convert_mat_matrix(filename, matname):
    """Convert a MATLAB matrix(matname) contained in a .mat file(filename)
    into a numpy.ndarray

    filename = name of .mat file with path, e.g. 'matrixfiles/mymatrix.mat'
    matname = name of matrix saved in .mat file, e.g. 'myMatrix'

    """
    mat_file = scipy.io.loadmat(filename)
    matrix = mat_file[matname]
    return matrix


wing_array = convert_mat_matrix('MATLAB_files/TreeNymphLeftWing.mat', 'Wing')
wing_array = wing_array.transpose()
wing_array = wing_array / 1000  # Convert units from mm to meter
wing_coords = {
    'x': wing_array[0],
    'y_le': wing_array[1],
    'y_te': wing_array[2]
}
treeNymphHead = buildbutterfly.BodyPart(0.02, 0.006, 0.006)
treeNymphThorax = buildbutterfly.BodyPart(0.1808, 0.01, 0.006)
treeNymphAbdomen = buildbutterfly.BodyPart(0.2404, 0.0289, 0.006)
treeNymphWing = buildbutterfly.Wing(wing_coords, 0.0587)
treeNymph = buildbutterfly.Butterfly(treeNymphHead, treeNymphThorax,
                                     treeNymphAbdomen, treeNymphWing)


output = open('treeNymph.pkl', 'wb')
pickle.dump(treeNymph, output, pickle.HIGHEST_PROTOCOL)
