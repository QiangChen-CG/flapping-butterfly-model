"""

DOCSTRING HERE

"""


class BodyPart(object):
    """Define a body part with a mass, length, and radius properties.  All
    body parts are modeled as spheroids of uniform density.

    mass = mass of the body part in grams(g), must be positive
    length = length of the body part in meters(m), cannot be negative
    radius = radius of the body part in meters(m), cannot be negative
    """

    def __init__(self, mass, length, radius):
        self.mass = mass  # grams
        self.length = length  # meters
        self.radius = radius  # meters

        # More pythonic way to implement these errors?
        if not (mass > 0):
            raise ValueError("mass must be positive")
        if length < 0:
            raise ValueError("length cannot be negative")
        if radius < 0:
            raise ValueError("radius cannot be negative")


class Wing(object):
    """DOCSTRING - change inputs to 3 1-D arrays, create dict in constructor"""
    def __init__(self, wingcoords, mass):
        self.wingcoords = wingcoords # x, y_le, y_te coords
        self.mass = mass # grams


class Butterfly(object):
    def __init__(self, head, thorax, abdomen, wing):
        self.head = head  # BodyPart class
        self.thorax = thorax  # BodyPart class
        self.abdomen = abdomen  # BodyPart class
        self.wing = wing  # Wing class


# TEST CODE, DELETE LATER

