import argparse
import warnings

import GPy
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
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


def log(out, args, optimal_hparams, final_error):
    out.writelines(f"Latitude: {args.lat}, Longitude: {args.lon},"
                   f"Number of months: {args.num_months}, Number of iterations: {args.num_iterations}\n"
                   f"Domain of lengthscale1: {args.domain_ls1}, lengthscale2: {args.domain_ls2}, lengthscale3: {args.domain_ls3}, "
                   f"Domain of variance1: {args.domain_v1}, variance2: {args.domain_v2}")

    out.writelines(f"ls1: {str(optimal_hparams[0])}, ls2: {str(optimal_hparams[1])}, "
                   f"ls3: {str(optimal_hparams[2])}, v1: {str(optimal_hparams[3])}, "
                   f"v2: {str(optimal_hparams[4])}, best_error: {final_error}\n")

    out.close()


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
        parameter hyperparameters: A vector containing the hyperparameters to optimise.
        return: The root mean squared error of the model.
    """

    # TODO: add support for multiple loss functions.
    rmse = 0
    print(f"Hyperparams shape: {hyperparameters.shape}")
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
        rmse += np.sqrt(np.square(y_all - y_pred).mean())

        # This used to be rmse += np.sum(all_y - y_pred)
        # which isn't the actual RMSE, but gave the positive definite error fewer times.

    print(f"RMSE: {rmse}")
    return rmse


def optimise(maximum_iterations=10, dom_tuples=None, acquisition_type="LCB", acquisition_weight=0.1):
    if dom_tuples is None:
        dom_tuples = [(0., 5.), (0., 1.), (0., 5.), (0., 1.), (0., 1.)]

    domain = [
        {'name': 'lengthscale1', 'type': 'continuous', 'domain': dom_tuples[0]},
        {'name': 'lengthscale2', 'type': 'continuous', 'domain': dom_tuples[1]},
        {'name': 'lengthscale3', 'type': 'continuous', 'domain': dom_tuples[2]},
        {'name': 'variance1', 'type': 'continuous', 'domain': dom_tuples[3]},
        {'name': 'variance2', 'type': 'continuous', 'domain': dom_tuples[4]}]

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

    kernel = GPy.kern.RBF(1, lengthscale=optimal_hparams[0], variance=optimal_hparams[3]) + \
             GPy.kern.StdPeriodic(1, lengthscale=optimal_hparams[1]) * \
             GPy.kern.PeriodicMatern32(1, lengthscale=optimal_hparams[2], variance=optimal_hparams[4])

    model = GPy.models.GPRegression(x_train.reshape(-1, 1), y_train.reshape(-1, 1), kernel=kernel, normalizer=True,
                                    noise_var=0.05)

    y_mean, y_std = model.predict(x_all.reshape(-1, 1))
    y_pred = draw_samples(y_mean, y_std)

    # Calculate RMSE — does it even make sense to use RMSE since
    # a GP doesn't make exact predictions, but specifies distributions
    # over functions?
    best_rmse = np.sqrt(np.square(y_all - y_pred).mean())

    return x_best, best_rmse, y_mean, y_std


def plot_fitted_model(x_train, x_test, x_all, y_train, y_test, y_mean, y_std, filename):
    plt.figure(figsize=(10, 5), dpi=100)
    plt.xlabel("time")
    plt.ylabel("precipitation")
    plt.scatter(x_train, y_train, lw=1, color="b", label="training dataset")
    plt.scatter(x_test, y_test, lw=1, color="r", label="testing dataset")
    plt.plot(x_all, y_mean, lw=3, color="g", label="GP mean")
    plt.fill_between(x_all, (y_mean + y_std).reshape(y_mean.shape[0]), (y_mean - y_std).reshape(y_mean.shape[0]),
                     facecolor="b", alpha=0.3, label="confidence")
    plt.legend(loc="upper left")
    plt.savefig(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the GPyOpt optimisation algorithm.')

    parser.add_argument("--lat", default=51.875, type=float,
                        help="The latitude of the point we want to investigate. Defaults to 51.875.")
    parser.add_argument("--lon", default=0.9375, type=float,
                        help="The longitude of the point we want to investigate. Defaults to 0.9375.")
    parser.add_argument("--num_months", default=1980, type=int,
                        help="How many months of data to consider, starting 01-01-1850. Defaults to 1980.")
    parser.add_argument("--training_size", default=0.8, type=float,
                        help="Percentage of the data to use for training. Defaults to 0.8.")
    parser.add_argument("--num_iterations", default=20, type=int,
                        help="Number of iterations of bayesian optimisation. Defaults to 20.")
    parser.add_argument("--domain_ls1", type=str,
                        help="The domain of the first hyperparameter (lengthscale1). Enter as start, finish")
    parser.add_argument("--domain_ls2", type=str,
                        help="The domain of the second hyperparameter (lengthscale2). Enter as start, finish")
    parser.add_argument("--domain_ls3", type=str,
                        help="The domain of the third hyperparameter (lengthscale3). Enter as start, finish")
    parser.add_argument("--domain_v1", type=str,
                        help="The domain of the fourth hyperparameter (variance1). Enter as start, finish")
    parser.add_argument("--domain_v2", type=str,
                        help="The domain of the fifth hyperparameter (variance2). Enter as start, finish")
    parser.add_argument("--acquisition_type", default="LCB", type=str,
                        help="Type of acquisition function to use in bayesian optimisation. Defaults to LCB.")
    parser.add_argument("--acquisition_weight", default=0.1, type=float,
                        help="The weight of the acquisition function. Defaults to 0.1")

    args = parser.parse_args()
    parsed_domain_tuples = []

    # TODO: unit test the tuple parsing.
    for domain in [args.domain_ls1, args.domain_ls2, args.domain_ls3, args.domain_v1, args.domain_v2]:
        if domain is not None:
            bounds = domain.split(",")
            parsed_domain_tuples.append((float(bounds[0]), float(bounds[1])))

    # Load the data.
    x_train, y_train, x_test, y_test, x_all, y_all = load_data(lat=args.lat,
                                                               lon=args.lon,
                                                               total_entries=args.num_months)

    optimal_hparams, best_error, y_mean, y_std = optimise(maximum_iterations=args.num_iterations,
                                                          dom_tuples=parsed_domain_tuples,
                                                          acquisition_type=args.acquisition_type,
                                                          acquisition_weight=args.acquisition_weight)

    # TODO: figure out a suitable filename.
    composite_filename = f"something"

    out = open(f"{composite_filename}.txt", "w")
    log(out, args, optimal_hparams, best_error)

    plot_fitted_model(x_train, x_test, x_all, y_train, y_test, y_mean, y_std, f"{composite_filename}.png")
