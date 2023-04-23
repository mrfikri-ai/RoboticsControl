import math
import matplotlib.pyplot as plt
import numpy as np

def f2d(x : np.ndarray, centers : np.ndarray):
    y = np.zeros(x.shape)
    c = 0.6 # standard deviation
    for center in centers:
        # Smooth distribution
        y = y + 5 * np.exp(-np.power(x - center, 2)/(2 * np.power(c,2)))
        # With noise
        # y = y + 5 * np.exp(-6 * np.power(x - center, 2)) + add_noise(np.copy(x))
    return y

def f3d(x : np.ndarray, y : np.ndarray):
    z = np.zeros((x.shape[0], y.shape[0]))
    p = np.array([[2, 0.25], [1, 2.25], [1.9, 1.9], [2.35, 1.25], [0.1, 0.1]])
    for point in p:
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                z[i][j] = z[i][j] + 5 * np.exp(-6 * (np.power(x[i] - point[0], 2) + np.power(y[j] - point[1], 2))) 
    
    # With the noise
    # z = z + add_noise(np.copy(z))
    return z 

def add_noise(x : np.ndarray):
    result = np.copy(x) + np.random.rand(x.shape[0]) * 0.2 - 0.1
    return result

def smooth_2d_by_anisotropic(array : np.ndarray, sigma : int) -> np.ndarray :
    def g(K : float, gradient_data : float) -> float :
        return 1 / (1 + math.pow(abs(gradient_data / K), 2))
    def smooth_helper(data : np.ndarray) -> np.ndarray :
        # value of K is 90% of total data. the paper recommends to use 90% of total gradient of the data
        K = 0
        for x in np.nditer(data):
            K += x
        K *= 0.9
        
        next_data = np.zeros(data.shape)
        for i in range(data.shape[0]):
            sum_of_constants_times_dData = 0
            # out-of-boundary pixel is assumed to have the same value as data[i, j]
            for di in [-1, 1]:
                if (0 <= i+di) and (i+di < data.shape[0]):
                    dData = data[i+di] - data[i]
                    constant = g(K, dData)
                    sum_of_constants_times_dData += constant * dData
            next_data[i] = data[i] + (0.5 * sum_of_constants_times_dData)
        return next_data

    data = np.copy(array)
    for i in range(sigma):
        data = smooth_helper(data)
    return data

def smooth_3d_by_anisotropic(array : np.ndarray, sigma : int) -> np.ndarray :
    def g(K : float, gradient_data : float) -> float :
        return 1 / (1 + (gradient_data / K)**2)
    def smooth_helper(data : np.ndarray) -> np.ndarray :
        # value of K is 90% of total data. the paper recommends to use 90% of total gradient of the data
        K = 0
        for x in np.nditer(data):
            K += x
        K *= 0.9

        next_data = np.zeros(data.shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                sum_of_constants_times_dData = 0
                # out-of-boundary pixel is assumed to have the same value as data[i, j]
                for di in [-1, 1]:
                    if (0 <= i+di) and (i+di < data.shape[0]):
                        dData = data[i+di, j] - data[i, j]
                        constant = g(K, dData)
                        sum_of_constants_times_dData += constant * dData
                for dj in [-1, 1]:
                    if (0 <= j+dj) and (j+dj < data.shape[1]):
                        dData = data[i, j+dj] - data[i, j]
                        constant = g(K, dData)
                        sum_of_constants_times_dData += constant * dData
                next_data[i, j] = data[i, j] + (0.25 * sum_of_constants_times_dData)
        return next_data

    data = np.copy(array)
    for i in range(sigma):
        data = smooth_helper(data)
    return data

def test_2d():
    x = np.linspace(-5, 10, 1001)
    y = f2d(x, np.array([0, 2, 4, 6, 8])) # x axis
    # y = f2d(x, np.array([0.25, 2.25, 1.9, 1.25, 0.1])) # y axis
    y1 = smooth_2d_by_anisotropic(y, sigma=50)
    y2 = smooth_2d_by_anisotropic(y, sigma=100)
    y3 = smooth_2d_by_anisotropic(y, sigma=200)
    y4 = smooth_2d_by_anisotropic(y, sigma=400)
    y5 = smooth_2d_by_anisotropic(y, sigma=1000)

    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    ax1 = fig.add_subplot(2, 3, 2)
    ax2 = fig.add_subplot(2, 3, 3)
    ax3 = fig.add_subplot(2, 3, 4)
    ax4 = fig.add_subplot(2, 3, 5)
    ax5 = fig.add_subplot(2, 3, 6)

    ax.plot(x, y)
    ax.set_title('sigma = 0')
    ax1.plot(x, y1)
    ax1.set_title('sigma = 50')
    ax2.plot(x, y2)
    ax2.set_title('sigma = 100')
    ax3.plot(x, y3)
    ax3.set_title('sigma = 200')
    ax4.plot(x, y4)
    ax4.set_title('sigma = 400')
    ax5.plot(x, y5)
    ax5.set_title('sigma = 1000')
    plt.show()

def test_3d():
    x = np.linspace(-1, 3, 101)
    y = np.linspace(-1, 3, 101)
    z = f3d(x, y)
    z1 = smooth_3d_by_anisotropic(z, sigma=2)
    z2 = smooth_3d_by_anisotropic(z, sigma=100)
    z3 = smooth_3d_by_anisotropic(z, sigma=200)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 4)

    def draw(axes : plt.Axes, z_value, title : str):
        axes.clabel(axes.contour(x, y, z_value), inline = True, fontsize = 8)
        axes.imshow(z_value, extent = [-1, 3, -1, 3], origin = 'lower', alpha = 0.5)
        axes.set_aspect('equal', 'box')
        axes.set_title(title)

    draw(ax, z, 'sigma = 0')
    draw(ax1, z1, 'sigma = 5')
    draw(ax2, z2, 'sigma = 10')
    draw(ax3, z3, 'sigma = 20')
    plt.show()

test_2d()
