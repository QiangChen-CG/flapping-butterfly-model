"""Converts data from MATLAB .mat files into usable python data.

This function extracts data in MATLAB's .mat file format and converts it
into usable python data(XXX EXPAND HERE - NUMPY ARRAY? ETC ETC)

"""


class BodyPart(object):
    """Define a body part with a mass, length, and radius properties.  All

    Body parts are modeled as spheroids of uniform density.

    mass = mass of the body part in grams(g)
    length = length of the body part in meters(m)
    radius = radius of the body part in meters(m)
    """
    def __init__(self, mass, length, radius):
        self.mass = mass  # grams
        self.length = length  # meters
        self.radius = radius  # meters

        # Check for negative property values and raise ValueError if found
        if not all(i >= 0 for i in [mass, length, radius]):
            raise ValueError("All property values must be positive or zero")


class Wing(object):
    def __init__(self):
        pass


class Butterfly(object):
    def __init__(self, head, thorax, abdomen, wing):
        self.head = head  # BodyPart class
        self.thorax = thorax  # BodyPart class
        self.abdomen = abdomen  # BodyPart class
        self.wing = wing  # Wing class

        if not all(type(i) is BodyPart for i in [head, thorax, abdomen]):
            raise TypeError("head, thorax, and abdomen must all be BodyPart "
                            "objects")
        if type(wing) is not Wing:
            raise TypeError("wing must be a Wing object")


treeNymphHead = BodyPart(0.02, 0.006, 0.006)
treeNymphThorax = BodyPart(0.1808, 0.01, 0.006)
treeNymphAbdomen = BodyPart(0.2404, 0.0289, 0.006)
treeNymphWing = Wing()
treeNymph = Butterfly(treeNymphHead, treeNymphThorax, treeNymphAbdomen,
                      treeNymphWing)

# TEST PRINTS - DELETE LATER
print(treeNymph.thorax.mass)
print("")
