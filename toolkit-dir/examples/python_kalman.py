import cv2
import numpy as np

from utils.tracker import Tracker
from utils.ex2_utils import get_patch
from utils.ex3_utils import create_cosine_window, create_gauss_peak


class KalmanTracker(Tracker):

    def __init__(self):
        self.parameters = KalmanParams()

    def name(self):
        return 'moss'

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor


        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        
        self.size = (int(region[2]), int(region[3]))
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)

        #self.template = get_patch(image, self.position, self.size)

        gaussian = create_gauss_peak((self.window, self.window), self.parameters.sigma)
        self.gaussian = np.fft.fft2(gaussian)

        patch, mask = get_patch(image, self.position, self.gaussian.shape)
        gray_image = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        self.cos_win = create_cosine_window(self.gaussian.shape)
        patch = self.cos_win * gray_image
        F = np.fft.fft2(patch)
        F_conjugate = np.conjugate(F)

        self.H = (self.gaussian * F_conjugate)/(F * F_conjugate + self.parameters.lam)

    def track(self, image):
        #print(f"Starting position: ({self.position[0]},{self.position[1]})")

        patch, mask = get_patch(image, self.position, self.gaussian.shape)
        gray_image = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        window = create_cosine_window(self.gaussian.shape)
        patch =  window * gray_image
        F_new = np.fft.fft2(patch)
        R = np.fft.ifft2(F_new * self.H)

        y,x = np.unravel_index(np.argmax(R), R.shape)
        if x > R.shape[1]/2:
            x = x-R.shape[1]
        if y > R.shape[0]/2:
            y = y-R.shape[0]
        
        self.position = (self.position[0] + x, self.position[1] + y)

        patch, mask = get_patch(image, self.position, self.gaussian.shape)
        gray_image = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch =  self.cos_win * gray_image
        F = np.fft.fft2(patch)
        F_conjugate = np.conjugate(F)

        H_current = (self.gaussian * F_conjugate)/(F * F_conjugate + self.parameters.lam)
        self.H = (1-self.parameters.alpha) * self.H + self.parameters.alpha * H_current

        return [self.position[0] - self.size[0]/2, self.position[1] - self.size[1]/2, self.size[0], self.size[1]]

class KalmanParams():

    def __init__(self, sigma=5, lam = 0.0001, enlarge_factor=1.2, alpha=0.05):
        self.lam = lam
        self.sigma = sigma
        self.enlarge_factor = enlarge_factor
        self.alpha = alpha
