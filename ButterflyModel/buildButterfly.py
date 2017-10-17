"""Converts data from MATLAB .mat files into usable python data.

This function extracts data in MATLAB's .mat file format and converts it
into usable python data(XXX EXPAND HERE - NUMPY ARRAY? ETC ETC)

"""

import os
# import scipy.io

# Test outputs, delete later
print(12)
print(os.path.dirname(__file__))


class BodyPart(object):
    def __init__(self, mass, length, radius):
        self.mass = mass #grams
        self.length = length #meters
        self.radius = radius #meters


class Butterfly(object):
    def __init__(self, head, thorax, abdomen, wing):
        self.head = head #BodyPart class
        self.thorax = thorax #BodyPart class
        self.abdomen = abdomen #BodyPart class
        self.wing = wing #BodyPart class?? Wing class??


treeNymphHead = BodyPart(0.02, 0.006, 0.006)
treeNymphThorax = BodyPart(0.1808, 0.01, 0.006)
treeNymphAbdomen = BodyPart(0.2404, 0.0289, 0.006)
treeNymph = Butterfly(treeNymphHead, treeNymphThorax, treeNymphAbdomen, 0)

print(treeNymph.thorax.mass)
print(treeNymph.abdomen.length)
