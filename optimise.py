import time
import warnings
import itertools

import GPy
import GPyOpt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from numpy.random import seed
from utils.utils import draw_samples, log, rms_error, parse_args

warnings.filterwarnings('ignore')
seed(12345)
sns.set()


# TODO: load global dataset instead of one single point. x[t] = [lat, lon, t]
def load_data(lat: float = 51.875, lon: float = 0.9375, total_entries: int = 1980, train_size: float = 0.8) -> tuple:
    """Load data from a .csv file with the specified parameters.
        :param lat: The latitude of the location.
        :param lon: The longitude of the location.
        :param total_entries: The total number of months to consider starting in Jan 1850.
        :param train_size: The proportion of the data to use for training.
        :return: A tuple containing the training set, the test set, as well as the two sets combined.
    """
    lat_str, lon_str = str(lat).replace(".", "_"), str(lon).replace(".", "_")
    pr_df = pd.read_csv(f"./data/{lat_str}x{lon_str}.csv")

    last_train_index = int(total_entries * train_size)
    last_validation_index = int(last_train_index + ((total_entries - last_train_index) // 3))

    training_set = pr_df.iloc[0:last_train_index]
    validation_set = pr_df.iloc[last_train_index:last_validation_index]
    test_set = pr_df.iloc[last_validation_index:total_entries]

    x_train = np.linspace(1, last_train_index,
                          last_train_index, dtype=np.int64)
    x_valid = np.linspace(last_train_index, last_validation_index,
                          last_validation_index - last_train_index, dtype=np.int64)
    x_test = np.linspace(last_validation_index, total_entries,
                         total_entries - last_validation_index, dtype=np.int64)
    x_all = np.concatenate((x_train, x_valid, x_test))

    # Renormalise precipitation measurements since they're all approximately r*10^-5
    y_train = (training_set["pr"].to_numpy()) * 10 ** 5
    y_valid = (validation_set["pr"].to_numpy()) * 10 ** 5
    y_test = test_set["pr"].to_numpy() * 10 ** 5
    y_all = np.concatenate([y_train, y_valid, y_test])

    return x_train, y_train, x_valid, y_valid, x_test, y_test, x_all, y_all


def fit_gp(hyperparameters: np.ndarray) -> float:
    """This function initialises a Gaussian process given :param hyperparameters,
        fits it to the data, and calculates the error of the test set predictions.
        :param hyperparameters: A vector containing the hyperparameters to optimise.
        :return: The total error of the model with these hyperparameters.
    """

    loss = 0
    y_mean = y_train.mean()
    for i in range(hyperparameters.shape[0]):

        # TODO: try a different model type, like sparseGP
        kernel = GPy.kern.RBF(1, lengthscale=hyperparameters[i, 0], variance=hyperparameters[i, 3]) + \
                 GPy.kern.StdPeriodic(1, lengthscale=hyperparameters[i, 1]) * \
                 GPy.kern.PeriodicMatern32(1, lengthscale=hyperparameters[i, 2], variance=hyperparameters[i, 4])
        model = GPy.models.GPRegression(x_train.reshape(-1, 1), (y_train-y_mean).reshape(-1, 1), kernel=kernel,
                                        normalizer=True, noise_var=0.05)

        # Here we must *only* predict on the validation set, not on all the values.
        # We want to tune the hyperparameters using just this data.
        y_mean, y_std = model.predict(x_valid.reshape(-1, 1))
        y_pred = draw_samples(y_mean, y_std)

        loss += rms_error(y_valid, y_pred)

    return loss


def grid_search_domains(k: int, maximum_iterations: int, model_type: str, initial_design_type: str,
                        acquisition_type: str, acquisition_weight: float, acquisition_optimiser_type: str,
                        composite_filename: str) -> None:
    """Implements grid search over the hyperparameter domains, then runs optimisation
        for each combination of domains.
        :param k: The number of values to consider for each hyperparameter.
        :param maximum_iterations: The maximum number of iterations to run the optimiser for.
        :param model_type: The type of model to use.
        :param initial_design_type: The type of initial design to use (random, latin hypercube).
        :param acquisition_type: The type of acquisition function to use.
        :param acquisition_weight: The weight of the acquisition function.
        :param acquisition_optimiser_type: The type of optimiser to use.
        :param composite_filename: The name of the file to save the results to.
    """

    # Generate hyperparameter domains on a log-scale grid.
    # The grid starts at 10^-3 and ends at 10^3.
    # :param k controls the number of values to generate.
    # Note that this means k^5 experiments are going to be run.
    # Generating e.g. 0.01 for a given hyperparameter means
    # its domain is going to be [0, 0.01].
    log_scales = [np.logspace(-3, 3, k) for _ in range(5)]

    out = open(f"{composite_filename}.txt", "a+")

    # Generate all possible combinations of hyperparameter domains.
    for combination in itertools.product(*log_scales):
        dom_tuples = [(0, val) for val in combination]

        # Run optimisation using the domains generated.
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
        plot_fitted_model(x_train, x_valid, x_test, x_all,
                          y_train, y_valid, y_test, y_mean, y_std,
                          f"{composite_filename}.png")
    out.close()


def optimise(maximum_iterations: int = 10, dom_tuples: list[tuple] = None, model_type: str = "GP",
             initial_design_type: str = "random", acquisition_type: str = "LCB", acquisition_weight: float = 0.1,
             acquisition_optimiser_type: str = "lbfgs") -> tuple:
    """Runs the optimisation algorithm for a given set of hyperparameter domains.
        Also takes in other hyperparameters for the optimisation procedure.
        :param maximum_iterations: The maximum number of iterations to run the optimiser for.
        :param dom_tuples: The hyperparameter domains to consider.
        :param model_type: The type of model to use.
        :param initial_design_type: The type of initial design to use (random, latin hypercube).
        :param acquisition_type: The type of acquisition function to use.
        :param acquisition_weight: The weight of the acquisition function.
        :param acquisition_optimiser_type: The type of optimiser to use.
        :return: The optimal hyperparameters, the best error, the mean and standard deviation of the predictions.
    """

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
    y_mean = y_train.mean()

    kernel = GPy.kern.RBF(1, lengthscale=optimal_hparams[0], variance=optimal_hparams[3]) + \
        GPy.kern.StdPeriodic(1, lengthscale=optimal_hparams[1]) * \
        GPy.kern.PeriodicMatern32(1, lengthscale=optimal_hparams[2], variance=optimal_hparams[4])

    model = GPy.models.GPRegression(x_train.reshape(-1, 1), (y_train-y_mean).reshape(-1, 1), kernel=kernel,
                                    normalizer=False, noise_var=0.05)

    # We are done optimising, so we can now make predictions
    # for the entire dataset, including the test set.
    y_mean, y_std = model.predict(x_all.reshape(-1, 1))
    y_pred = draw_samples(y_mean, y_std)

    best_rmse = rms_error(y_all, y_pred)

    return optimal_hparams, best_rmse, y_mean, y_std


def plot_fitted_model(x_train: np.ndarray, x_valid: np.ndarray, x_test: np.ndarray, x_all: np.ndarray,
                      y_train: np.ndarray, y_valid: np.ndarray, y_test: np.ndarray, y_mean: np.ndarray,
                      y_std: np.ndarray, filename) -> None:
    """Plots a fitted GP model's predictions.
        :param x_train: The training set months.
        :param x_valid: The validation set months.
        :param x_test: The test set months.
        :param x_all: The full training + test set.
        :param y_train: The training precipitation values.
        :param y_test: The test precipitation values.
        :param y_valid: The validation precipitation values.
        :param y_mean: The mean of the predictions.
        :param y_std: The standard deviation of the predictions.
        :param filename: The filename to save the plot to.
    """
    plt.figure(figsize=(10, 5), dpi=100)
    plt.xlabel("time")
    plt.ylabel("precipitation")
    plt.scatter(x_train, y_train-y_train.mean(), lw=1, color="b", label="training dataset")
    plt.scatter(x_valid, y_valid-y_valid.mean(), lw=1, color="y", label="validation dataset")
    plt.scatter(x_test, y_test-y_test.mean(), lw=1, color="r", label="testing dataset")
    plt.plot(x_all, y_mean, lw=3, color="g", label="GP mean")
    plt.fill_between(x_all, (y_mean + y_std).reshape(y_mean.shape[0]), (y_mean - y_std).reshape(y_mean.shape[0]),
                     facecolor="b", alpha=0.3, label="confidence")
    plt.legend(loc="upper left")
    plt.savefig(filename)


if __name__ == '__main__':
    args = parse_args()

    # Load the data.
    x_train, y_train, x_test, y_test, x_valid, y_valid, x_all, y_all = load_data(lat=args.lat,
                                                                                 lon=args.lon,
                                                                                 total_entries=args.num_months)

    composite_filename = f"./logs/{args.model_type}_{args.acquisition_type}_{args.acquisition_optimiser_type}_" \
                         f"{str(args.lat).replace('.', 'p')}_{str(args.lon).replace('.', 'p')}_" \
                         f"{args.num_iterations}_{args.num_months}"

    # Here the flow of the program splits into two scenarios:
    # Run optimisation with domains found by grid search or
    # with domains specified by the user as command-line arguments.
    if args.grid_search:
        grid_search_domains(k=args.k,
                            maximum_iterations=args.num_iterations,
                            model_type=args.model_type,
                            initial_design_type=args.initial_design,
                            acquisition_type=args.acquisition_type,
                            acquisition_weight=args.acquisition_weight,
                            acquisition_optimiser_type=args.acquisition_optimiser_type,
                            composite_filename=composite_filename, )
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
                                                              acquisition_optimiser_type=args.acquisition_optimiser_type
                                                              )
        toc = time.perf_counter()

        out = open(f"{composite_filename}.txt", "a+")
        log(out, args, optimal_hparams, best_error, parsed_domain_tuples, toc - tic)
        out.close()
        plot_fitted_model(x_train, x_valid, x_test, x_all,
                          y_train, y_valid, y_test, y_mean, y_std,
                          f"{composite_filename}.png")
