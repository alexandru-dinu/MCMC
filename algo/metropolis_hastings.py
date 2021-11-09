import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

mus = np.array([5, 5])
sigmas = np.array([[1, 0.9], [0.9, 1]])


def circle(x, y):
    return (x - 1) ** 2 + (y - 2) ** 2 - 3 ** 2


def pgauss(x, y):
    return stats.multivariate_normal.pdf([x, y], mean=mus, cov=sigmas)


def metropolis_hastings(p, num_iter=1000):
    x, y = 0.0, 0.0
    samples = np.zeros((num_iter, 2))

    for i in range(num_iter):
        eps = np.random.normal(0, 1, size=2)
        x_star, y_star = np.array([x, y]) + eps
        if np.random.rand() < p(x_star, y_star) / p(x, y):
            x, y = x_star, y_star
        samples[i] = [x, y]

    return samples


if __name__ == "__main__":
    samples = metropolis_hastings(circle, num_iter=10000)
    sns.jointplot(samples[:, 0], samples[:, 1])
    plt.title("Circle")
    plt.show()

    samples = metropolis_hastings(pgauss, num_iter=10000)
    sns.jointplot(samples[:, 0], samples[:, 1])
    plt.title("Gauss")
    plt.show()
