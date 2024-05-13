import cv2
import numpy as np
import sympy as sp
from ex2_utils import Tracker, get_patch, create_epanechnik_kernel, extract_histogram
from ex3_utils import create_cosine_window, create_gauss_peak
from ex4_utils import sample_gauss

def RW(q, r):
    F = [
        [0,0],
        [0,0]
    ]
    L = [
        [1,0],
        [0,1]
    ]
    H = np.array([
        [1,0],
        [0,1]
    ], dtype=np.float32)

    R = r * np.array([
        [1,0],
        [0,1]
    ], dtype=np.float32)

    T = sp.symbols('T')
    F = sp.Matrix(F)
    L = sp.Matrix(L)
    Fi = np.array(sp.exp(F*T).subs(T,1), dtype=np.float32)
    Q = np.array(sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T)).subs(T,1), dtype=np.float32)
    return Fi, Q, H, R


def NCV(q,r):
    F = np.array([
        [0,0,1,0],
        [0,0,0,1],
        [0,0,0,0],
        [0,0,0,0]
    ])
    L = np.array([
        [0,0],
        [0,0],
        [1,0],
        [0,1]
    ])
    H = np.array([
        [1,0,0,0],
        [0,1,0,0]
    ], dtype=np.float32)

    R = r * np.array([
        [1,0],
        [0,1]
    ], dtype=np.float32)

    T = sp.symbols('T')
    F = sp.Matrix(F)
    L = sp.Matrix(L)
    Fi = np.array(sp.exp(F*T).subs(T,1), dtype=np.float32)
    Q = np.array(sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T)).subs(T,1), dtype=np.float32)
    return Fi, Q, H, R

def NCA(q,r):
    F = [
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
    ]
    L = [
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [1,0],
        [0,1]
    ]
    H = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0]
    ], dtype=np.float32)

    R = r * np.array([
        [1,0],
        [0,1]
    ], dtype=np.float32)

    T = sp.symbols('T')
    F = sp.Matrix(F)
    L = sp.Matrix(L)
    Fi = np.array(sp.exp(F*T).subs(T,1), dtype=np.float32)
    Q = np.array(sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T)).subs(T,1), dtype=np.float32)
    return Fi, Q, H, R


class KalmanTracker(Tracker):

    def __init__(self):
        self.parameters = KalmanParams()
        self.algs = {"RW": RW, "NCV": NCV, "NCA": NCA}

    def name(self):
        return 'kalman'

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3])


        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        
        self.size = (int(region[2]), int(region[3]))
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)

        
        self.Fi, self.Q, _, _ = self.algs[self.parameters.model](self.parameters.q, self.parameters.r)


        kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)
        patch, mask = get_patch(image, self.position, (kernel.shape[1], kernel.shape[0]))
        self.histogram = extract_histogram(patch, nbins=self.parameters.nbins, weights=kernel)
        self.means = [self.position[0], self.position[1]]
        if self.parameters.model == "NCV":
            self.means = [self.position[0], self.position[1], 0, 0]
        elif self.parameters.model == "NCA":
            self.means = [self.position[0], self.position[1], 0, 0, 0, 0]

        self.samples = sample_gauss(self.means, self.Q, self.parameters.n)
        print(self.samples.shape)
        self.weights = np.ones(self.parameters.n)

    def track(self, image):
        #print(f"Starting position: ({self.position[0]},{self.position[1]})")

        self.weights = np.cumsum(self.weights/np.sum(self.weights))
        rand_samples = np.random.rand(self.parameters.n, 1)
        sampled_idxs = np.digitize(rand_samples, self.weights)
        self.samples = self.samples[sampled_idxs.flatten(), :]

        self.samples = np.matmul(self.Fi, self.samples.T)
        self.samples += sample_gauss(np.zeros(self.means), self.Q, self.parameters.n)
        print(self.samples.shape)

        for i in range (len(self.samples)):
            a = 3

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
    def __init__(self, sigma=5, nbins=16, model="RW", q=1, r=1, n=100, enlarge_factor=1.2, alpha=0.05):
        self.model = model
        self.q = q
        self.r = r
        self.n = n
        self.sigma = sigma
        self.nbins = nbins
        self.enlarge_factor = enlarge_factor
        self.alpha = alpha
