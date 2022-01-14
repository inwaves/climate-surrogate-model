import GPy
import GPyOpt
import numpy as np
import warnings
import matplotlib.pyplot as plt

from numpy.random import seed
from scipy.stats import multivariate_normal as mvn

warnings.filterwarnings('ignore')
seed(12345)

# TODO: add support for multiple models
# TODO: add support for multiple datasets
# TODO: add support for multiple optimizers
# TODO: add support for multiple loss functions

def draw_samples(y_mean, y_std):
    """Make some predictions by drawing from GP distributions."""
    mean = y_mean.reshape(-1)
    cov = np.diag(y_std.reshape(-1))
    y_pred = mvn.rvs(mean, cov, 1)

    return y_pred


def fit_gp(x):
    """This is the function that GPyOpt will optimise.
    :parameter x: A vector containing the hyperparameters to optimise.
    :return: The root mean squared error of the model.
    """

    rmse = 0
    for i in range(x.shape[0]):
        # Create and train an SVR on this data.
        kernel = GPy.kern.RBF(1, lengthscale=x[i, 0], variance=x[i, 3]) + GPy.kern.StdPeriodic(1, lengthscale=x[
            i, 1]) * GPy.kern.PeriodicMatern32(1, lengthscale=x[i, 2], variance=x[i, 4])
        model = GPy.models.GPRegression(x_train.reshape(-1, 1), y_train.reshape(-1, 1), kernel=kernel, normalizer=True,
                                        noise_var=0.05)
        y_mean, y_std = model.predict(x_all.reshape(-1, 1))
        y_pred = draw_samples(y_mean, y_std)

        # Calculate RMSE — does it even make sense to use RMSE since
        # a GP doesn't make exact predictions, but specifies distributions
        # over functions?
        all_y = np.concatenate([y_train, y_test])
        rmse += np.sum(all_y - y_pred)

    return rmse

def load_data():
    pass

def train(maximum_iterations=10, acquisition_type="LCB", acquisition_weight=0.1):
    domain = [
        {'name': 'lengthscale1', 'type': 'continuous', 'domain': (0., 5.)},
        {'name': 'lengthscale2', 'type': 'continuous', 'domain': (0., 1.)},
        {'name': 'lengthscale3', 'type': 'continuous', 'domain': (0., 5.)},
        {'name': 'variance1', 'type': 'continuous', 'domain': (0., 1.)},
        {'name': 'variance2', 'type': 'continuous', 'domain': (0., 1.)}]

    opt = GPyOpt.methods.BayesianOptimization(f=fit_gp,  # function to optimize
                                              domain=domain,  # box-constraints of the problem
                                              acquisition_type=acquisition_type,
                                              acquisition_weight=acquisition_weight)

    # Optimise the hyperparameters.
    opt.run_optimization(max_iter=maximum_iterations)
    opt.plot_convergence() # TODO: get these out to a file.

    # Get the optimised hyperparameters.
    x_best = opt.X[np.argmin(opt.Y)]
    print(
        f"ls1: {str(x_best[0])}, ls2: {str(x_best[1])}, ls3: {str(x_best[2])}, v1: {str(x_best[3])}, v2: {str(x_best[4])}\n")

    # TODO: get the optimised loss value, log to file.
    # TODO: add logging -  image with composite filename, and log file.

def plot_fitted_model(x_train=None, x_best=None, y_train=None, x_all=None, x_test=None, y_test=None):
    kernel = GPy.kern.RBF(1, lengthscale=x_best[0], variance=x_best[2]) + GPy.kern.PeriodicMatern32(1,
                                                                                                    lengthscale=x_best[
                                                                                                        1],
                                                                                                    variance=x_best[3])
    model = GPy.models.GPRegression(x_train.reshape(-1, 1), y_train.reshape(-1, 1), kernel=kernel, normalizer=True,
                                    noise_var=0.05)
    y_mean, y_std = model.predict(x_all.reshape(-1, 1))

    # TODO: Get this out to a file.
    plt.figure(figsize=(10, 5), dpi=100)
    plt.xlabel("time")
    plt.ylabel("precipitation")
    plt.scatter(x_train, y_train, lw=1, color="b", label="training dataset")
    plt.scatter(x_test, y_test, lw=1, color="r", label="testing dataset")
    plt.plot(x_all, y_mean, lw=3, color="g", label="GP mean")
    plt.fill_between(x_all, (y_mean + y_std).reshape(y_mean.shape[0]), (y_mean - y_std).reshape(y_mean.shape[0]),
                     facecolor="b", alpha=0.3, label="confidence")
    plt.legend(loc="upper left")
    plt.show()

if __name__=='__main__':
    # TODO: parse arguments

