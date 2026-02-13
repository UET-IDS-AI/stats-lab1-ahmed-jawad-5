import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    values = np.random.normal(loc=0, scale=1, size=n)
    plt.hist(values, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Normal Distribution")
    plt.show()
    return values

def uniform_histogram(n):
    values = np.random.uniform(low=0, high=10, size=n)
    plt.hist(values, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Uniform Distribution")
    plt.show()
    return values

def bernoulli_histogram(n):
    values = np.random.binomial(n=1, p=0.5, size=n)
    plt.hist(values, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Benoulli Histogram")
    plt.show()
    return values


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    arr = np.asarray(data)
    return np.sum(arr) / arr.size

def sample_variance(data):
    arr = np.asarray(data)
    avg = sample_mean(arr)
    squared_diff = (arr - avg) ** 2
    return np.sum(squared_diff) / (arr.size - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    arr = np.sort(np.asarray(data))
    n = arr.size

    minimum = arr[0]
    maximum = arr[-1]

    mid = n // 2

    # Median and halves
    if n % 2 == 0:
        median = (arr[mid - 1] + arr[mid]) / 2
        lower_half = arr[:mid]
        upper_half = arr[mid:]
    else:
        median = arr[mid]
        lower_half = arr[:mid + 1]
        upper_half = arr[mid:]

    # Q1
    m1 = len(lower_half)
    idx1 = m1 // 2
    if m1 % 2 == 0:
        q1 = (lower_half[idx1 - 1] + lower_half[idx1]) / 2
    else:
        q1 = lower_half[idx1]

    # Q3
    m2 = len(upper_half)
    idx2 = m2 // 2
    if m2 % 2 == 0:
        q3 = (upper_half[idx2 - 1] + upper_half[idx2]) / 2
    else:
        q3 = upper_half[idx2]

    return (minimum, maximum, median, q1, q3)


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    arr_x = np.asarray(x)
    arr_y = np.asarray(y)

    mean_x = sample_mean(arr_x)
    mean_y = sample_mean(arr_y)

    product_diff = (arr_x - mean_x) * (arr_y - mean_y)
    return np.sum(product_diff) / (len(arr_x) - 1)


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    variance_x = sample_variance(x)
    variance_y = sample_variance(y)
    covariance_xy = sample_covariance(x, y)

    matrix = np.array([[variance_x, covariance_xy],
                       [covariance_xy, variance_y]])
    return matrix

