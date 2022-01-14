import GPy
import GPyOpt
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd

from numpy.random import seed
from scipy.stats import multivariate_normal as mvn

warnings.filterwarnings('ignore')
seed(12345)


# TODO: add support for multiple models
# TODO: add support for multiple optimizers

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

    # TODO: add support for multiple loss functions
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
        rmse += np.sum(y_all - y_pred)

    return rmse


def load_data(lat=51.875, lon=0.9375, train_size=0.8):
    lat_str, lon_str = str(lat), str(lon)
    pr_df = pd.read_csv(f"{lat_str}_{lon_str}.csv")

    last_train_index = pr_df.size * train_size

    training_set = pr_df.iloc[0:last_train_index]
    test_set = pr_df.iloc[last_train_index:pr_df.size]

    x_train = np.linspace(1, last_train_index, last_train_index)
    x_test = np.linspace(last_train_index, pr_df.size, pr_df.size - last_train_index)
    x_all = np.concatenate((x_train, x_test))

    # Renormalise precipitation measurements since they're all approximately r*10^-5
    y_train = (training_set["pr"].to_numpy()) * 10 ** 5
    y_test = test_set["pr"].to_numpy() * 10 ** 5
    y_all = np.concatenate([y_train, y_test])

    return x_train, y_train, x_test, y_test, x_all, y_all


def optimise(maximum_iterations=10, acquisition_type="LCB", acquisition_weight=0.1):
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
    opt.plot_convergence()  # TODO: get these out to a file.

    # Get the optimised hyperparameters.
    x_best = opt.X[np.argmin(opt.Y)]
    print(
        f"ls1: {str(x_best[0])}, ls2: {str(x_best[1])}, ls3: {str(x_best[2])}, v1: {str(x_best[3])}, v2: {str(x_best[4])}\n")
    # TODO: get the optimised loss value, log to file.
    # TODO: add logging -  image with composite filename, and log file.

    return x_best


def plot_fitted_model(x_tr, optimal_hparams, y_tr, x_all_p, x_te, y_te):
    kernel = GPy.kern.RBF(1, lengthscale=optimal_hparams[0], variance=optimal_hparams[3]) + \
             GPy.kern.StdPeriodic(1, lengthscale=optimal_hparams[1]) * \
             GPy.kern.PeriodicMatern32(1, lengthscale=optimal_hparams[2], variance=optimal_hparams[4])

    model = GPy.models.GPRegression(x_tr.reshape(-1, 1), y_tr.reshape(-1, 1), kernel=kernel, normalizer=True,
                                    noise_var=0.05)
    y_mean, y_std = model.predict(x_all_p.reshape(-1, 1))

    # TODO: Get this out to a file.
    plt.figure(figsize=(10, 5), dpi=100)
    plt.xlabel("time")
    plt.ylabel("precipitation")
    plt.scatter(x_tr, y_tr, lw=1, color="b", label="training dataset")
    plt.scatter(x_te, y_te, lw=1, color="r", label="testing dataset")
    plt.plot(x_all_p, y_mean, lw=3, color="g", label="GP mean")
    plt.fill_between(x_all_p, (y_mean + y_std).reshape(y_mean.shape[0]), (y_mean - y_std).reshape(y_mean.shape[0]),
                     facecolor="b", alpha=0.3, label="confidence")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == '__main__':
    # TODO: parse arguments

    # Load the data.
    x_train, y_train, x_test, y_test, x_all, y_all = load_data()
    optimal_hparams = optimise()
    plot_fitted_model(x_train, optimal_hparams, y_train, x_all, x_test, y_test)
