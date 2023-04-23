import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def f2d(x : np.ndarray, centers : np.ndarray):
    y = np.zeros(x.shape)
    c = np.sqrt(1/12)
    for center in centers:
        y = y + 5 * np.exp(np.power(x - center, 2)/(2 * np.power(c,2)))
        # y = y + 5 * np.exp(-6 * np.power(x - center, 2))
        # y = y + np.exp(-0.5 * np.power(x - center, 2)) + np.exp(-0.5 * np.power(x - center + 10, 2))
    return y

def f3d(x : np.ndarray, y : np.ndarray):
    z = np.zeros((x.shape[0], y.shape[0]))
    p = np.array([[2, 0.25], [1, 2.25], [1.9, 1.9], [2.35, 1.25], [0.1, 0.1]])
    for point in p:
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                z[i][j] = z[i][j] + 5 * np.exp(-6 * (np.power(x[i] - point[0], 2) + np.power(y[j] - point[1], 2)))
    return z

def add_noise(x : np.ndarray):
    result = np.copy(x) + np.random.rand(x.shape[0]) * 10 - 0.1
    return result

def test_2d():
    x = np.linspace(-10, 10, 1001)
    # y = f2d(x, np.array([2, 1, 1.9, 2.35, 0.1])) # x axis
    y = f2d(x, np.array([0.25, 1.25, 2.9, 3.25, 4.1])) # y axis
    y1 = gaussian_filter(y, sigma=20)
    y2 = gaussian_filter(y, sigma=40)
    y3 = gaussian_filter(y, sigma=60)
    y4 = gaussian_filter(y, sigma=80)
    y5 = gaussian_filter(y, sigma=100)

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
    ax1.set_title('sigma = 20')
    ax2.plot(x, y2)
    ax2.set_title('sigma = 40')
    ax3.plot(x, y3)
    ax3.set_title('sigma = 60')
    ax4.plot(x, y4)
    ax4.set_title('sigma = 80')
    ax5.plot(x, y5)
    ax5.set_title('sigma = 100')
    plt.show()

def test_3d():
    x = np.linspace(-1, 3, 101)
    y = np.linspace(-1, 3, 101)
    z = f3d(x, y)
    z1 = gaussian_filter(z, sigma=2)
    z2 = gaussian_filter(z, sigma=10)
    z3 = gaussian_filter(z, sigma=20)

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
    draw(ax1, z1, 'sigma = 2')
    draw(ax2, z2, 'sigma = 10')
    draw(ax3, z3, 'sigma = 20')
    plt.show()

test_2d()
