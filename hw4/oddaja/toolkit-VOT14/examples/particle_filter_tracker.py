import numpy as np
import sympy as sp
from enum import Enum
import cv2
from matplotlib import pyplot as plt
from utils.ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram
from utils.ex4_utils import sample_gauss
from utils.tracker import Tracker

class models(Enum):
    RW = 'RW'
    NCV = 'NCV'
    NCA = 'NCA'
    
class colors(Enum):
    HSV = cv2.COLOR_BGR2HSV
    LAB = cv2.COLOR_BGR2LAB
    RGB = cv2.COLOR_BGR2RGB
    YCRCB = cv2.COLOR_BGR2YCR_CB

def prepare_matrices(model:models, q):
    # Calculate Fi and Q matrices based on type of motion model
    F, L, = None, None
    if model == models.RW:
        F = [[0, 0],
              [0, 0]]
        
        L = [[1, 0],
             [0, 1]]
        
    elif model == models.NCV:
        F = [[0, 0, 1, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        
        L = [[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]]
        
        
    elif model == models.NCA:
        F = [[0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]
        
        L = [[0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]]
        
    else:
        print(f'Unknown model type!')
        exit(-1)

    T = sp.symbols('T')
    F = sp.Matrix(F)
    L = sp.Matrix(L)
    Fi = np.array(sp.exp(F*T).subs(T, 1), dtype= float)
    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T)).subs(T, 1)
    Q = np.array(Q, dtype=float)

    return Fi, Q


def plot_particles(img, particles, position):
    plt.imshow(img)
    plt.scatter(particles[0], particles[1], s = 2, color='green')
    plt.scatter(position[0], position[1], color = 'red')
    plt.waitforbuttonpress()
    plt.close()


class Parameters():
    def __init__(self, sigma = 0.5, histogram_bins = 16, model_type: models = models.NCV,
                 n_particles = 100, dist_sigma = 1, alpha = 0.05, enlarge_factor = 2,
                 color: colors = None):
        self.sigma = sigma
        self.histogram_bins = histogram_bins
        self.model_type = model_type
        self.n_particles = n_particles
        self.dist_sigma = dist_sigma
        self.alpha = alpha
        self.enlarge_factor = enlarge_factor
        self.color = color

class ParticleFilterTracker(Tracker):

    def __init__(self):
        self.parameters = Parameters()

    def name(self):
        return 'ParticleFilterTracker'
    

    def initialize(self, img, region):
        # Chenge color space
        if self.parameters.color != None:
            img = cv2.cvtColor(img, self.parameters.color.value)
    
        region = [int(x) for x in region]
        if region[2] % 2 == 0:
            region[2] += 1
        if region[3] % 2 == 0:
            region[3] += 1

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        # Locate patch and location of searched area
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        self.size = (region[2], region[3])        
        
        #Calculate dynamic parameters
        image_pl = img.shape[0] * img.shape[1]
        patch_pl = self.size[0] * self.size[1]
        self.parameters.q = self.q = max(0, int(patch_pl / image_pl * 200))
        #self.parameters.q =100
            
        # Define motion model
        self.system_matrix, self.system_covariance = prepare_matrices(self.parameters.model_type, self.parameters.q)
        self.epanechnik = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)
        self.patch_size = self.epanechnik.shape[::-1]
        self.patch, _ = get_patch(img, self.position, self.size)
        self.histogram = extract_histogram(self.patch, self.parameters.histogram_bins, self.epanechnik)

        # Handle different motion model particle sizes
        self.particle_state = [self.position[0], self.position[1]]
        if self.parameters.model_type == models.NCV:
            self.particle_state.extend([0, 0])
            
        if self.parameters.model_type == models.NCA:
            self.particle_state.extend([0, 0, 0, 0])

        # Normilze initial weights
        self.particles = sample_gauss(self.particle_state, self.system_covariance, self.parameters.n_particles)
        self.weights = np.array([1 / self.parameters.n_particles for _ in range(self.parameters.n_particles)])
        

    def track(self, img):
        # Change color space
        if self.parameters.color != None:
            img = cv2.cvtColor(img, self.parameters.color._value_)
            
        # Sampling
        weights_cumsumed = np.cumsum(self.weights)
        rand_samples = np.random.rand(self.parameters.n_particles, 1)
        sampled_indexes = np.digitize(rand_samples, weights_cumsumed)
        particles_new = self.particles[sampled_indexes.flatten(), :]
        
        # Add noise
        noise = sample_gauss([0 for _ in range(self.system_matrix.shape[0])], self.system_covariance, self.parameters.n_particles)
        self.particles = np.transpose(np.matmul(self.system_matrix, np.transpose(particles_new))) + noise
        
        h, w = img.shape[0], img.shape[1]
        particles_x, particles_y = [], []
        
        # Iterate trough particles and calculate new weights/probablities
        for i, _ in enumerate(particles_new):
            x = self.particles[i][0]
            y = self.particles[i][1]
            
            # Set weights to 0 if particle out of fram
            if x < 0 or  y < 0 or x > w or y > h:
                self.weights[i] = 0
                continue
            
            particles_x.append(x)
            particles_y.append(y)
            patch_new, _= get_patch(img, (x, y), self.patch_size)
            hist_new = extract_histogram(patch_new, self.parameters.histogram_bins, weights=self.epanechnik)
            h_dist = 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(hist_new) - np.sqrt(self.histogram))
            self.weights[i] = np.exp(-0.5 * h_dist ** 2 / self.parameters.dist_sigma ** 2)    

        # If sum of all weights is 0, set equal weights to all of them
        if np.sum(self.weights) == 0:
            self.weights += 0.000001
        self.weights = self.weights / (np.sum(self.weights))
            
        # Calculate new position
        new_x = np.sum([particle[0] * self.weights[i] for i, particle in enumerate(self.particles)])
        new_y = np.sum([particle[1] * self.weights[i] for i, particle in enumerate(self.particles)])
        # if new_x > w:
        #     x = w
        # if new_x < 0:
        #     x = 0
        # if new_y > h:
        #     new_y = h
        # if new_y < 0:
        #     new_y =0

        # Update template histogram with new detection
        self.position = (new_x, new_y)
        #plot_particles(img, [particles_x, particles_y], self.position)
        self.patch, _ = get_patch(img, (new_x, new_y), self.patch_size)
        if self.patch.shape[:1] != self.epanechnik.shape:
            h, w = self.epanechnik.shape
            self.patch = self.patch[:h, :w, :]
        hist_new_template = extract_histogram(self.patch, self.parameters.histogram_bins, weights=self.epanechnik)
        self.histogram = (1 - self.parameters.alpha) * self.histogram + self.parameters.alpha* hist_new_template

        return [new_x - self.size[0] / 2, new_y - self.size[1] / 2, self.size[0], self.size[1]]