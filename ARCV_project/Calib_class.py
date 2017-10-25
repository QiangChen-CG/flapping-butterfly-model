import cv2
import numpy as np
import pickle
from collections import namedtuple


# HSV_params = namedtuple('HSV_params', 'hist_low, hist_hi, h_ext, s_ext, v_ext')

class HSVCalibration:
    def __init__(self, color_name):
        if type(color_name) is not str:
            raise TypeError('Input is not a string')
        self.color = color_name

    # def SetHistParams(self, params):
    #     self.

    def __mouseCallFunc(self, event, x, y, flags, params):
        if (event == cv2.EVENT_LBUTTONDOWN and not params['drag']):
            params['point1'] = (x, y)
            params['drag'] = 1
        elif (event == cv2.EVENT_MOUSEMOVE and params['drag']):
            img1 = params['frame'].copy()
            cv2.rectangle(img1, params['point1'], (x, y), 255, 1, 8, 0)
            cv2.imshow('camera', img1)
        elif (event == cv2.EVENT_LBUTTONUP and params['drag']):
            if (x, y) != params['point1']:
                params['roi'] = params['frame'][params['point1'][1]: y, params['point1'][0]: x]
                params['roiSelected'] = 1
            params['drag'] = 0

    def GetROI(self, c):
        mouseCallParams = {'drag': 0, 'roiConfirmed': 0, 'roi': None, 'point1': None, 'frame': None}
        firstPass = 1

        while not mouseCallParams['roiConfirmed']:
            _, frame = c.read()
            cv2.imshow('camera', frame)

            mouseCallParams['frame'] = frame
            cv2.setMouseCallback('camera', self.__mouseCallFunc, mouseCallParams);

            # if mouseCallParams['roiSelected']:
            if mouseCallParams['roi'] is not None:
                roi = mouseCallParams['roi']
                if firstPass:
                    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    roi_h = cv2.split(roi_hsv)[0]
                    roi_s = cv2.split(roi_hsv)[1]
                    roi_v = cv2.split(roi_hsv)[2]

                    cv2.destroyWindow('crop')
                    cv2.destroyWindow('roi_hue')
                    cv2.destroyWindow('roi_sat')
                    cv2.destroyWindow('roi_val')

                    cv2.imshow('crop', roi)
                    cv2.imshow('roi hue', roi_h)
                    cv2.imshow('roi sat', roi_s)
                    cv2.imshow('roi val', roi_v)

                    cv2.moveWindow('crop', frame.shape[1] + 18 * 1 + 20, 20)
                    cv2.moveWindow('roi hue', frame.shape[1] + roi.shape[1] + 18 * 2 + 20, 20)
                    cv2.moveWindow('roi sat', frame.shape[1] + 18 * 1 + 20, roi.shape[0] + 62)
                    cv2.moveWindow('roi val', frame.shape[1] + roi.shape[1] + 18 * 2 + 20, roi.shape[0] + 62)
                    firstPass = 0

            # press ESC to break loop
            if cv2.waitKey(5) == 27:
                break

        return mouseCallParams['roi']

    def ConfirmROI(self, c, roi):
        _, frame = c.read()  # read and display current frame
        cv2.imshow('camera', frame)
        cv2.moveWindow('camera', 20, 20)

        pct_low = 5
        pct_hi = 95
        h_ext = 5
        s_ext = 5
        v_ext = 5

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_h = cv2.split(roi_hsv)[0]
        roi_s = cv2.split(roi_hsv)[1]
        roi_v = cv2.split(roi_hsv)[2]

        cv2.imshow('crop', roi)
        cv2.imshow('roi hue', roi_h)
        cv2.imshow('roi sat', roi_s)
        cv2.imshow('roi val', roi_v)
        # cv2.imshow('colorhist', h)

        cv2.moveWindow('crop', frame.shape[1] + 18 * 1 + 20, 20)
        cv2.moveWindow('roi hue', frame.shape[1] + roi.shape[1] + 18 * 2 + 20, 20)
        cv2.moveWindow('roi sat', frame.shape[1] + 18 * 1 + 20, roi.shape[0] + 62)
        cv2.moveWindow('roi val', frame.shape[1] + roi.shape[1] + 18 * 2 + 20, roi.shape[0] + 62)

        while 1:
            _, frame = c.read()  # read and display current frame
            cv2.imshow('camera', frame)

            if cv2.waitKey(5) == 27:
                cv2.destroyWindow('crop')
                cv2.destroyWindow('roi hue')
                cv2.destroyWindow('roi sat')
                cv2.destroyWindow('roi val')
                cv2.destroyWindow('colorhist')
                HSVvalues1 = [np.percentile(roi_h, pct_low), np.percentile(roi_h, pct_hi),
                              np.percentile(roi_s, pct_low), np.percentile(roi_s, pct_hi),
                              np.percentile(roi_v, pct_low), np.percentile(roi_v, pct_hi)]

                HSVvalues = np.zeros(6)
                HSVvalues[0] = np.amax((0, HSVvalues1[0] - h_ext * (HSVvalues1[1] - HSVvalues1[0])))
                HSVvalues[1] = np.amin((180, HSVvalues1[1] + h_ext * (HSVvalues1[1] - HSVvalues1[0])))
                HSVvalues[2] = np.amax((0, HSVvalues1[2] - s_ext * (HSVvalues1[3] - HSVvalues1[2])))
                HSVvalues[3] = np.amin((255, HSVvalues1[3] + s_ext * (HSVvalues1[3] - HSVvalues1[2])))
                HSVvalues[4] = np.amax((0, HSVvalues1[4] - v_ext * (HSVvalues1[5] - HSVvalues1[4])))
                HSVvalues[5] = np.amin((255, HSVvalues1[5] + v_ext * (HSVvalues1[5] - HSVvalues1[4])))
                HSVvalues = np.int32(np.around(HSVvalues))
                break

        return HSVvalues

    def Calibrate(self, cam):
        # begin video capture
        c = cv2.VideoCapture(cam)
        cv2.namedWindow('camera')
        cv2.moveWindow('camera', 20, 20)
        roiConfirmed = 0

        while not roiConfirmed:
            roi = self.GetROI(c)
            HSVvalues = self.ConfirmROI(c, roi)
            roiConfirmed = 1
