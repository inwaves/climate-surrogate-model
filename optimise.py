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


def draw_samples(y_mean, y_std):
    """Make some predictions by drawing from GP distributions."""
    mean = y_mean.reshape(-1)
    cov = np.diag(y_std.reshape(-1))
    y_pred = mvn.rvs(mean, cov, 1)

    return y_pred


def log(optimal_hparams, final_error):
    print(f"ls1: {str(optimal_hparams[0])}, ls2: {str(optimal_hparams[1])}, "
          f"ls3: {str(optimal_hparams[2])}, v1: {str(optimal_hparams[3])}, "
          f"v2: {str(optimal_hparams[4])}\n")
    pass


def load_data(lat=51.875, lon=0.9375, total_entries=1980, train_size=0.8):
    lat_str, lon_str = str(lat).replace(".", ""), str(lon).replace(".", "")
    pr_df = pd.read_csv(f"{lat_str}_{lon_str}.csv")

    last_train_index = int(total_entries * train_size)

    training_set = pr_df.iloc[0:last_train_index]
    test_set = pr_df.iloc[last_train_index:total_entries]

    x_train = np.linspace(1, last_train_index, last_train_index, dtype=np.int64)
    x_test = np.linspace(last_train_index, total_entries, total_entries - last_train_index, dtype=np.int64)
    x_all = np.concatenate((x_train, x_test))

    # Renormalise precipitation measurements since they're all approximately r*10^-5
    y_train = (training_set["pr"].to_numpy()) * 10 ** 5
    y_test = test_set["pr"].to_numpy() * 10 ** 5
    y_all = np.concatenate([y_train, y_test])

    return x_train, y_train, x_test, y_test, x_all, y_all


def fit_gp(hyperparameters):
    """This is the function that GPyOpt will optimise.
    :parameter hyperparameters: A vector containing the hyperparameters to optimise.
    :return: The root mean squared error of the model.
    """

    # TODO: add support for multiple loss functions.
    rmse = 0
    for i in range(hyperparameters.shape[0]):
        kernel = GPy.kern.RBF(1, lengthscale=hyperparameters[i, 0], variance=hyperparameters[i, 3]) + \
                 GPy.kern.StdPeriodic(1, lengthscale=hyperparameters[i, 1]) * \
                 GPy.kern.PeriodicMatern32(1, lengthscale=hyperparameters[i, 2], variance=hyperparameters[i, 4])
        model = GPy.models.GPRegression(x_train.reshape(-1, 1), y_train.reshape(-1, 1), kernel=kernel, normalizer=True,
                                        noise_var=0.05)
        y_mean, y_std = model.predict(x_all.reshape(-1, 1))
        y_pred = draw_samples(y_mean, y_std)

        # Calculate RMSE — does it even make sense to use RMSE since
        # a GP doesn't make exact predictions, but specifies distributions
        # over functions?
        rmse += np.sum(y_all - y_pred)

    return rmse


def optimise(maximum_iterations=10, acquisition_type="LCB", acquisition_weight=0.1):
    domain = [
        {'name': 'lengthscale1', 'type': 'continuous', 'domain': (0., 5.)},
        {'name': 'lengthscale2', 'type': 'continuous', 'domain': (0., 1.)},
        {'name': 'lengthscale3', 'type': 'continuous', 'domain': (0., 5.)},
        {'name': 'variance1', 'type': 'continuous', 'domain': (0., 1.)},
        {'name': 'variance2', 'type': 'continuous', 'domain': (0., 1.)}]

    # TODO: add support for multiple models
    # TODO: add support for multiple optimizers

    opt = GPyOpt.methods.BayesianOptimization(f=fit_gp,  # function to optimize
                                              domain=domain,  # box-constraints of the problem
                                              acquisition_type=acquisition_type,
                                              acquisition_weight=acquisition_weight)

    # Optimise the hyperparameters.
    opt.run_optimization(max_iter=maximum_iterations)
    opt.plot_convergence()  # TODO: get these out to a file.

    # Get the optimised hyperparameters.
    x_best = opt.X[np.argmin(opt.Y)]
    log(x_best, final_error)

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
    x_train, y_train, x_test, y_test, x_all, y_all = load_data(total_entries=100)
    optimal_hparams = optimise(maximum_iterations=30)
    plot_fitted_model(x_train, optimal_hparams, y_train, x_all, x_test, y_test)
