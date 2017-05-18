import math

import data

def dot_kf(u, v):
    """
    The basic dot product kernel returns u*v+1.

    Args:
        u: list of numbers
        v: list of numbers

    Returns:
        dot(u,v) + 1
    """
    res = 0
    for i in range(0, len(u)):
        res += float(u[i]) * float(v[i])
    return res + 1

def poly_kernel(d):
    """
    The polynomial kernel.

    Args:
        d: a number

    Returns:
        A function that takes two vectors u and v,
        and returns (u*v+1)^d.
    """
    def kf(u, v):
        return (dot_kf(u, v))**d
    return kf

def exp_kernel(s):
    """
    The exponential kernel.

    Args:
        s: a number

    Returns:
        A function that takes two vectors u and v,
        and returns exp(-||u-v||/(2*s^2))
    """
    def kf(u, v):
        sum = 0
        for i in range(len(u)):
            sum += (u[i] - v[i])**2
        return math.exp(-math.sqrt(sum) / (2*(s**2)))
    return kf

class Perceptron(object):

    def __init__(self, kf):
        """
        Args:
            kf - a kernel function that takes in two vectors and returns
            a single number.
        """
        self.kf = kf
        # TODO: add more fields as needed
        self.mistake_points = [];
        self.mistake_labels = [];

    def update(self, point, label):
        """
        Updates the parameters of the perceptron, given a point and a label.

        Args:
            point: a list of numbers
            label: either 1 or -1

        Returns:
            True if there is an update (prediction is wrong),
            False otherwise (prediction is accurate).
        """
        if self.predict(point) == label:
            return True
        else:
            self.mistake_points.append(point)
            self.mistake_labels.append(label)
            return False

    def predict(self, point):
        """
        Given a point, predicts the label of that point (1 or -1).
        """
        total = 0
        for i in range(0, len(self.mistake_points)):
            total += self.mistake_labels[i] * self.kf(self.mistake_points[i], point)
        if total > 0:
            return 1
        else:
            return -1

# Feel free to add any helper functions as needed.


if __name__ == '__main__':
    val_data, val_labs = data.load_data('data/validation.csv')
    test_data, test_labs = data.load_data('data/test.csv')
    # TODO: implement code for running the problems
    x = Perceptron(dot_kf)
    d = [1,3,5,7,10,15,20]
    z = Perceptron(exp_kernel(10))
    # for j in d:
    y = Perceptron(poly_kernel(5))
    for i in range(0, len(test_data)):
        y.update(test_data[i], test_labs[i])
        z.update(test_data[i], test_labs[i])
        if (i > 0) and (i % 100 == 0):
            lossY = float(len(y.mistake_points)) / i
            lossZ = float(len(z.mistake_points)) / i
            print i, ": ", lossY
            print i, ": ", lossZ
    print i, ": ", float(len(y.mistake_points)) / 1000
    print i, ": ", float(len(z.mistake_points)) / 1000

