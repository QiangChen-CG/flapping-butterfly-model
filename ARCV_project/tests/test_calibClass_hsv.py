import os
import Calib_class

# Makes parent directory the working directory
os.chdir('..')

HSV_blue = Calib_class.HSVCalibration('blue')

HSV_blue.Calibrate(0)