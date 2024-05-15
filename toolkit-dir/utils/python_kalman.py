import cv2
import numpy as np
import sympy as sp
from ex2_utils import Tracker, get_patch, create_epanechnik_kernel, extract_histogram
from ex4_utils import sample_gauss

def RW(q):
    F = [
        [0,0],
        [0,0]
    ]
    L = [
        [1,0],
        [0,1]
    ]
    
    T = sp.symbols('T')
    F = sp.Matrix(F)
    L = sp.Matrix(L)
    Fi = np.array(sp.exp(F*T).subs(T,1), dtype=np.float32)
    Q = np.array(sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T)).subs(T,1), dtype=np.float32)
    return Fi, Q


def NCV(q):
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

    T = sp.symbols('T')
    F = sp.Matrix(F)
    L = sp.Matrix(L)
    Fi = np.array(sp.exp(F*T).subs(T,1), dtype=np.float32)
    Q = np.array(sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T)).subs(T,1), dtype=np.float32)
    return Fi, Q

def NCA(q):
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

    T = sp.symbols('T')
    F = sp.Matrix(F)
    L = sp.Matrix(L)
    Fi = np.array(sp.exp(F*T).subs(T,1), dtype=np.float32)
    Q = np.array(sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T)).subs(T,1), dtype=np.float32)
    return Fi, Q


class KalmanTracker(Tracker):

    def hellinger(self, h):
        return np.sqrt(np.sum((np.sqrt(h)-np.sqrt(self.histogram))**2))/np.sqrt(2)
    
    def new_weight(self, distance):
        return np.exp(-(distance**2)/(2*(self.parameters.hell_sig**2)))

    def set_q(self, image):
        area1 = self.size[0]*self.size[1]
        area2 = image.shape[0]*image.shape[1]
        return 100*area1/area2

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

        #self.window = max(region[2], region[3])
        
        self.size = (int(region[2]), int(region[3]))
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)

        q = self.set_q(image)
        self.Fi, self.Q = self.algs[self.parameters.model](q)

        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)
        patch, mask = get_patch(image, self.position, (self.kernel.shape[1], self.kernel.shape[0]))
        self.histogram = extract_histogram(patch, nbins=self.parameters.nbins, weights=self.kernel)
        self.means = [self.position[0], self.position[1]]
        if self.parameters.model == "NCV":
            self.means = [self.position[0], self.position[1], 0, 0]
        elif self.parameters.model == "NCA":
            self.means = [self.position[0], self.position[1], 0, 0, 0, 0]

        self.samples = sample_gauss(self.means, self.Q, self.parameters.n)
        self.weights = np.ones(self.parameters.n)
        self.weights /= np.sum(self.weights)
        #print(self.weights)
        #print(self.samples[:, 1])
        #print(self.weights*self.samples[:, 0])
        #print(np.sum(self.weights*self.samples[:, 0]))
        pos_x = np.sum(self.weights*self.samples[:, 0])
        pos_y = np.sum(self.weights*self.samples[:, 1])
        #print(f"Official position: ({self.position[0]},{self.position[1]})")
        #print(f"Particle position: ({pos_x},{pos_y})")

    def track(self, image):
        #print(f"Starting position: ({self.position[0]},{self.position[1]})")

        self.weights = np.cumsum(self.weights)
        rand_samples = np.random.rand(self.parameters.n, 1)
        sampled_idxs = np.digitize(rand_samples, self.weights)
        self.samples = self.samples[sampled_idxs.flatten(), :]


        self.samples = np.matmul(self.Fi, self.samples.T).T
        self.samples += sample_gauss(np.zeros(len(self.means)), self.Q, self.parameters.n)

        for i in range(len(self.samples)):
            x = self.samples[i][0]
            y = self.samples[i][1]
            if x < 0 or y < 0 or x > image.shape[1] or y > image.shape[0]:
                self.weights[i] = 0
                continue
            patch, mask = get_patch(image, (x, y), (self.kernel.shape[1], self.kernel.shape[0]))
            current_histogram = extract_histogram(patch, nbins=self.parameters.nbins, weights=self.kernel)
            distance = self.hellinger(current_histogram)
            self.weights[i] = self.new_weight(distance)
        
        self.weights /= np.sum(self.weights)
        pos_x = np.sum(self.weights*self.samples[:, 0])
        pos_y = np.sum(self.weights*self.samples[:, 1])
        #print(f"Update position: ({pos_x},{pos_y})")
        patch, mask = get_patch(image, (pos_x, pos_y), (self.kernel.shape[1], self.kernel.shape[0]))
        self.histogram = (1-self.parameters.alpha)*self.histogram + self.parameters.alpha*extract_histogram(patch, nbins=self.parameters.nbins, weights=self.kernel)
        self.position = (pos_x, pos_y)
        return [self.position[0] - self.size[0]/2, self.position[1] - self.size[1]/2, self.size[0], self.size[1]]

class KalmanParams():
    def __init__(self, sigma=0.5, nbins=16, model="NCV", n=100, hell_sig=1, alpha=0.05):
        self.model = model
        self.n = n
        self.hell_sig = hell_sig
        self.sigma = sigma
        self.nbins = nbins
        self.alpha = alpha
