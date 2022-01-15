import argparse
import time
import warnings
import itertools

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


def log(file_out, exp_args, hparams, final_error, domains, time_elapsed):
    """Log the results of the optimisation."""
    file_out.writelines(f"----------------------------------------------------\n"
                        f"Experiment took: {time_elapsed:.2f} seconds\n")
    file_out.writelines(f"Experiment parameters:\n"
                        f"Latitude: {exp_args.lat}, Longitude: {exp_args.lon}\n"
                        f"Number of months: {exp_args.num_months}, Number of iterations: {exp_args.num_iterations}\n"
                        f"Domain of lengthscale1: {domains[0]}, lengthscale2: {domains[1]}, "
                        f"lengthscale3: {domains[2]} \n "
                        f"Domain of variance1: {domains[3]}, variance2: {domains[4]}\n")

    file_out.writelines(f"Experiment results:\n"
                        f"ls1: {str(hparams[0])}, ls2: {str(hparams[1])}, "
                        f"ls3: {str(hparams[2])}, v1: {str(hparams[3])}, "
                        f"v2: {str(hparams[4])}, best_error: {final_error}\n")


def load_data(lat=51.875, lon=0.9375, total_entries=1980, train_size=0.8):
    lat_str, lon_str = str(lat).replace(".", "_"), str(lon).replace(".", "_")
    pr_df = pd.read_csv(f"./data/{lat_str}x{lon_str}.csv")

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


def rms_error(y_true, y_pred) -> np.ndarray:
    """Calculate the root mean squared error."""
    # return np.sqrt(np.square(y_true - y_pred).mean())
    return np.sum(y_true - y_pred)


def fit_gp(hyperparameters):
    """This is the function that GPyOpt will optimise.
        parameter hyperparameters: A vector containing the hyperparameters to optimise.
        return: The root mean squared error of the model.
    """

    loss = 0
    for i in range(hyperparameters.shape[0]):
        kernel = GPy.kern.RBF(1, lengthscale=hyperparameters[i, 0], variance=hyperparameters[i, 3]) + \
                 GPy.kern.StdPeriodic(1, lengthscale=hyperparameters[i, 1]) * \
                 GPy.kern.PeriodicMatern32(1, lengthscale=hyperparameters[i, 2], variance=hyperparameters[i, 4])
        model = GPy.models.GPRegression(x_train.reshape(-1, 1), y_train.reshape(-1, 1), kernel=kernel, normalizer=True,
                                        noise_var=0.05)
        y_mean, y_std = model.predict(x_all.reshape(-1, 1))
        y_pred = draw_samples(y_mean, y_std)

        loss += rms_error(y_all, y_pred)

    return loss


def grid_search_domains(maximum_iterations, model_type, initial_design_type, acquisition_type,
                        acquisition_weight, acquisition_optimiser_type, composite_filename):
    """Implement grid search over the hyperparameter domains.
    This should let you run multiple optimisation experiments and log them."""
    log_scales = [np.logspace(-3, 3, 2) for _ in range(5)]
    out = open(f"{composite_filename}.txt", "a+")
    for combination in itertools.product(*log_scales):
        dom_tuples = [(0, val) for val in combination]
        tic = time.perf_counter()
        optimal_hparams, best_error, y_mean, y_std = optimise(maximum_iterations=maximum_iterations,
                                                              dom_tuples=dom_tuples,
                                                              model_type=model_type,
                                                              initial_design_type=initial_design_type,
                                                              acquisition_type=acquisition_type,
                                                              acquisition_weight=acquisition_weight,
                                                              acquisition_optimiser_type=acquisition_optimiser_type)
        toc = time.perf_counter()

        log(out, args, optimal_hparams, best_error, dom_tuples, toc - tic)
        plot_fitted_model(x_train, x_test, x_all, y_train, y_test, y_mean, y_std, f"{composite_filename}.png")
    out.close()


def optimise(maximum_iterations=10, dom_tuples=None, model_type="GP", initial_design_type="random",
             acquisition_type="LCB", acquisition_weight=0.1, acquisition_optimiser_type="lbfgs") -> tuple:
    if dom_tuples is None:
        dom_tuples = [(0., 5.), (0., 1.), (0., 5.), (0., 1.), (0., 1.)]

    domain = [
        {'name': 'lengthscale1', 'type': 'continuous', 'domain': dom_tuples[0]},
        {'name': 'lengthscale2', 'type': 'continuous', 'domain': dom_tuples[1]},
        {'name': 'lengthscale3', 'type': 'continuous', 'domain': dom_tuples[2]},
        {'name': 'variance1', 'type': 'continuous', 'domain': dom_tuples[3]},
        {'name': 'variance2', 'type': 'continuous', 'domain': dom_tuples[4]}]

    opt = GPyOpt.methods.BayesianOptimization(f=fit_gp,  # function to optimize
                                              domain=domain,  # box-constraints of the problem
                                              model_type=model_type,  # model type
                                              initial_design_type=initial_design_type,  # initial design
                                              acquisition_type=acquisition_type,  # acquisition function
                                              acquisition_weight=acquisition_weight,
                                              acquisition_optimizer_type=acquisition_optimiser_type)

    # Optimise the hyperparameters.
    opt.run_optimization(max_iter=maximum_iterations)

    # To plot optimisation details uncomment this:
    # opt.plot_convergence(filename="optimisation_details.png")

    # Get the optimised hyperparameters.
    optimal_hparams = opt.X[np.argmin(opt.Y)]

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

    return optimal_hparams, best_rmse, y_mean, y_std


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

    parser.add_argument("--acquisition_type", default="LCB", type=str,
                        help="Type of acquisition function to use in bayesian optimisation. "
                             "Defaults to LCB. Choose from EI, MPI, LCB.")
    parser.add_argument("--acquisition_weight", default=0.1, type=float,
                        help="The weight of the acquisition function. Defaults to 0.1")
    parser.add_argument("--acquisition_optimiser_type", default="lbfgs", type=str,
                        help="Which optimiser to use for the acquisition function. "
                             "Defaults to L-BFGS. Choose from L-BFGS, DIRECT, CMA.")
    parser.add_argument("--domain_ls1", default="0,5", type=str,
                        help="The domain of the first hyperparameter (lengthscale1). Enter as start, finish")
    parser.add_argument("--domain_ls2", default="0,1", type=str,
                        help="The domain of the second hyperparameter (lengthscale2). Enter as start, finish")
    parser.add_argument("--domain_ls3", default="0,5", type=str,
                        help="The domain of the third hyperparameter (lengthscale3). Enter as start, finish")
    parser.add_argument("--domain_v1", default="0,1", type=str,
                        help="The domain of the fourth hyperparameter (variance1). Enter as start, finish")
    parser.add_argument("--domain_v2", default="0,1", type=str,
                        help="The domain of the fifth hyperparameter (variance2). Enter as start, finish")
    parser.add_argument("--grid_search", default=False, type=bool,
                        help="Whether to perform a grid search over the domains of the hyperparameters. ")
    parser.add_argument("--initial_design", default="random", type=str,
                        help="The type of initial design, where to collect points. "
                             "Defaults to random. Choose from random, latin.")
    parser.add_argument("--lat", default=51.875, type=float,
                        help="The latitude of the point we want to investigate. Defaults to 51.875.")
    parser.add_argument("--lon", default=0.9375, type=float,
                        help="The longitude of the point we want to investigate. Defaults to 0.9375.")
    parser.add_argument("--model_type", default="GP", type=str,
                        help="The type of model to use. Defaults to GP. Choose from: GP, sparseGP, warpedGP, RF.")
    parser.add_argument("--num_iterations", default=20, type=int,
                        help="Number of iterations of bayesian optimisation. Defaults to 20.")
    parser.add_argument("--num_months", default=1980, type=int,
                        help="How many months of data to consider, starting 01-01-1850. Defaults to 1980.")
    parser.add_argument("--training_size", default=0.8, type=float,
                        help="Percentage of the data to use for training. Defaults to 0.8.")

    args = parser.parse_args()


    # Load the data.
    x_train, y_train, x_test, y_test, x_all, y_all = load_data(lat=args.lat,
                                                               lon=args.lon,
                                                               total_entries=args.num_months)

    composite_filename = f"./logs/{args.model_type}_{args.acquisition_type}_{args.acquisition_optimiser_type}_" \
                         f"{str(args.lat).replace('.', 'p')}_{str(args.lon).replace('.', 'p')}_" \
                         f"{args.num_iterations}_{args.num_months}"

    if args.grid_search:
        grid_search_domains(maximum_iterations=args.num_iterations,
                            model_type=args.model_type,
                            initial_design_type=args.initial_design,
                            acquisition_type=args.acquisition_type,
                            acquisition_weight=args.acquisition_weight,
                            acquisition_optimiser_type=args.acquisition_optimiser_type,
                            composite_filename=composite_filename,)
    else:
        parsed_domain_tuples = []

        for dom in [args.domain_ls1, args.domain_ls2, args.domain_ls3, args.domain_v1, args.domain_v2]:
            if dom is not None:
                bounds = dom.split(",")
                parsed_domain_tuples.append((float(bounds[0]), float(bounds[1])))

        tic = time.perf_counter()
        optimal_hparams, best_error, y_mean, y_std = optimise(maximum_iterations=args.num_iterations,
                                                              dom_tuples=parsed_domain_tuples,
                                                              model_type=args.model_type,
                                                              initial_design_type=args.initial_design,
                                                              acquisition_type=args.acquisition_type,
                                                              acquisition_weight=args.acquisition_weight,
                                                              acquisition_optimiser_type=args.acquisition_optimiser_type)
        toc = time.perf_counter()

        out = open(f"{composite_filename}.txt", "a+")
        log(out, args, optimal_hparams, best_error, parsed_domain_tuples, toc - tic)
        out.close()
        plot_fitted_model(x_train, x_test, x_all, y_train, y_test, y_mean, y_std, f"{composite_filename}.png")
