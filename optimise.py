import time
import warnings
import itertools

import GPy
import GPyOpt
import seaborn as sns
import numpy as np

from numpy.random import seed
from utils.utils import draw_samples, log, rms_error, parse_args, load_data_single_coordinate, plot_model_predictions

warnings.filterwarnings('ignore')
seed(12345)
sns.set()


def fit_gp_k1(hyperparameters: np.ndarray) -> float:
    """This function initialises a Gaussian process given :param hyperparameters,
        fits it to the data, and calculates the error of the test set predictions.
        :param hyperparameters: A vector containing the hyperparameters to optimise.
        :return: The total error of the model with these hyperparameters.
    """

    loss = 0
    y_mean = y_train.mean()
    for i in range(hyperparameters.shape[0]):
        # TODO: try a different model type, like sparseGP
        kernel = GPy.kern.RBF(1, lengthscale=hyperparameters[i, 0], variance=hyperparameters[i, 4]) + \
                 (GPy.kern.RBF(1, lengthscale=hyperparameters[i, 1], variance=hyperparameters[i, 5]) *
                  GPy.kern.Cosine(1, lengthscale=hyperparameters[i, 2], variance=hyperparameters[i, 6])) + \
                 GPy.kern.sde_RatQuad(1, lengthscale=hyperparameters[i, 3], variance=hyperparameters[i, 7])
        model = GPy.models.GPRegression(x_train.reshape(-1, 1), (y_train - y_mean).reshape(-1, 1), kernel=kernel,
                                        normalizer=True, noise_var=0.05)

        # Here we must *only* predict on the validation set, not on all the values.
        # We want to tune the hyperparameters using just this data.
        y_mean, y_std = model.predict(x_valid.reshape(-1, 1))
        y_pred = draw_samples(y_mean, y_std)

        loss += rms_error(y_valid, y_pred)

    return loss


def fit_gp_k2(hyperparameters: np.ndarray) -> float:
    """This function initialises a Gaussian process given :param hyperparameters,
        fits it to the data, and calculates the error of the test set predictions.
        :param hyperparameters: A vector containing the hyperparameters to optimise.
        :return: The total error of the model with these hyperparameters.
    """

    loss = 0
    y_mean = y_train.mean()
    for i in range(hyperparameters.shape[0]):
        # TODO: try a different model type, like sparseGP
        kernel = GPy.kern.RBF(1, lengthscale=hyperparameters[i, 0], variance=hyperparameters[i, 5]) + \
                 (GPy.kern.RBF(1, lengthscale=hyperparameters[i, 1], variance=hyperparameters[i, 6]) *
                  GPy.kern.Cosine(1, lengthscale=hyperparameters[i, 2], variance=hyperparameters[i, 7])) + \
                 GPy.kern.sde_RatQuad(1, lengthscale=hyperparameters[i, 3], variance=hyperparameters[i, 8]) + \
                 (GPy.kern.RBF(1, lengthscale=hyperparameters[i, 4], variance=hyperparameters[i, 9]) *
                  GPy.kern.sde_White(1, variance=hyperparameters[i, 10]))
        model = GPy.models.GPRegression(x_train.reshape(-1, 1), (y_train - y_mean).reshape(-1, 1), kernel=kernel,
                                        normalizer=True, noise_var=0.05)

        # Here we must *only* predict on the validation set, not on all the values.
        # We want to tune the hyperparameters using just this data.
        y_mean, y_std = model.predict(x_valid.reshape(-1, 1))
        y_pred = draw_samples(y_mean, y_std)

        loss += rms_error(y_valid, y_pred)

    return loss


def grid_search_domains(k: int, kernel: int, maximum_iterations: int, model_type: str, initial_design_type: str,
                        acquisition_type: str, acquisition_weight: float, acquisition_optimiser_type: str,
                        composite_filename: str) -> None:
    """Implements grid search over the hyperparameter domains, then runs optimisation
        for each combination of domains.
        :param k: The number of values to consider for each hyperparameter.
        :param kernel: Kernel 1 is RBF-Cosine, kernel 2 is RBF-Cosine-Noise.
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
    # Note that this means k^num_parameters experiments are going to be run.
    # Generating e.g. 0.01 for a given hyperparameter means
    # its domain is going to be [0, 0.01].
    num_hparams = 8 if kernel == 1 else 11
    log_scales = [np.logspace(-3, 3, k) for _ in range(num_hparams)]

    out = open(f"{composite_filename}.txt", "a+")

    # Generate all possible combinations of hyperparameter domains.
    for combination in itertools.product(*log_scales):
        dom_tuples = [(0, val) for val in combination]

        # Run optimisation using the domains generated.
        tic = time.perf_counter()
        optimal_hparams, best_error, y_mean, y_std = optimise(maximum_iterations=maximum_iterations,
                                                              kernel_choice=args.kernel,
                                                              dom_tuples=dom_tuples, model_type=model_type,
                                                              initial_design_type=initial_design_type,
                                                              acquisition_type=acquisition_type,
                                                              acquisition_weight=acquisition_weight,
                                                              acquisition_optimiser_type=acquisition_optimiser_type)
        toc = time.perf_counter()

        log(out, kernel, args, optimal_hparams, best_error, dom_tuples, toc - tic)
        plot_model_predictions(x_train, x_valid, x_test, x_all,
                               y_train, y_valid, y_test, y_mean, y_std,
                          f"{composite_filename}.png")
    out.close()


def optimise(maximum_iterations: int = 10, kernel_choice: int = 1, dom_tuples: list[tuple] = None,
             model_type: str = "GP", initial_design_type: str = "random", acquisition_type: str = "LCB",
             acquisition_weight: float = 0.1, acquisition_optimiser_type: str = "lbfgs") -> tuple:
    """Runs the optimisation algorithm for a given set of hyperparameter domains.
        Also takes in other hyperparameters for the optimisation procedure.
        :param kernel_choice: Kernel 1 is RBF-Cosine, kernel 2 is RBF-Cosine-Noise.
        :param maximum_iterations: The maximum number of iterations to run the optimiser for.
        :param dom_tuples: The hyperparameter domains to consider.
        :param model_type: The type of model to use.
        :param initial_design_type: The type of initial design to use (random, latin hypercube).
        :param acquisition_type: The type of acquisition function to use.
        :param acquisition_weight: The weight of the acquisition function.
        :param acquisition_optimiser_type: The type of optimiser to use.
        :return: The optimal hyperparameters, the best error, the mean and standard deviation of the predictions.
    """
    if kernel_choice == 1:
        domain = [
            {'name': 'lengthscale1', 'type': 'continuous', 'domain': dom_tuples[0]},
            {'name': 'lengthscale2', 'type': 'continuous', 'domain': dom_tuples[1]},
            {'name': 'lengthscale3', 'type': 'continuous', 'domain': dom_tuples[2]},
            {'name': 'lengthscale4', 'type': 'continuous', 'domain': dom_tuples[3]},
            {'name': 'variance1', 'type': 'continuous', 'domain': dom_tuples[4]},
            {'name': 'variance2', 'type': 'continuous', 'domain': dom_tuples[5]},
            {'name': 'variance3', 'type': 'continuous', 'domain': dom_tuples[6]},
            {'name': 'variance4', 'type': 'continuous', 'domain': dom_tuples[7]}]

        opt = GPyOpt.methods.BayesianOptimization(f=fit_gp_k1,  # function to optimize
                                                  domain=domain,  # box-constraints of the problem
                                                  model_type=model_type,  # model type
                                                  initial_design_type=initial_design_type,  # initial design
                                                  acquisition_type=acquisition_type,  # acquisition function
                                                  acquisition_weight=acquisition_weight,
                                                  acquisition_optimizer_type=acquisition_optimiser_type)

    else:
        domain = [
            {'name': 'lengthscale1', 'type': 'continuous', 'domain': dom_tuples[0]},
            {'name': 'lengthscale2', 'type': 'continuous', 'domain': dom_tuples[1]},
            {'name': 'lengthscale3', 'type': 'continuous', 'domain': dom_tuples[2]},
            {'name': 'lengthscale4', 'type': 'continuous', 'domain': dom_tuples[3]},
            {'name': 'lengthscale5', 'type': 'continuous', 'domain': dom_tuples[4]},
            {'name': 'variance1', 'type': 'continuous', 'domain': dom_tuples[5]},
            {'name': 'variance2', 'type': 'continuous', 'domain': dom_tuples[6]},
            {'name': 'variance3', 'type': 'continuous', 'domain': dom_tuples[7]},
            {'name': 'variance4', 'type': 'continuous', 'domain': dom_tuples[8]},
            {'name': 'variance5', 'type': 'continuous', 'domain': dom_tuples[9]},
            {'name': 'variance6', 'type': 'continuous', 'domain': dom_tuples[10]}]

        opt = GPyOpt.methods.BayesianOptimization(f=fit_gp_k2,  # function to optimize
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

    if kernel_choice == 1:
        kernel = GPy.kern.RBF(1, lengthscale=optimal_hparams[0], variance=optimal_hparams[4]) + \
                 (GPy.kern.RBF(1, lengthscale=optimal_hparams[1], variance=optimal_hparams[5]) *
                  GPy.kern.Cosine(1, lengthscale=optimal_hparams[2], variance=optimal_hparams[6])) + \
                 GPy.kern.sde_RatQuad(1, lengthscale=optimal_hparams[3], variance=optimal_hparams[7])
    else:
        kernel = GPy.kern.RBF(1, lengthscale=optimal_hparams[0], variance=optimal_hparams[5]) + \
                 (GPy.kern.RBF(1, lengthscale=optimal_hparams[1], variance=optimal_hparams[6]) *
                  GPy.kern.Cosine(1, lengthscale=optimal_hparams[2], variance=optimal_hparams[7])) + \
                 GPy.kern.sde_RatQuad(1, lengthscale=optimal_hparams[3], variance=optimal_hparams[8]) + \
                 (GPy.kern.RBF(1, lengthscale=optimal_hparams[4], variance=optimal_hparams[9]) *
                  GPy.kern.sde_White(1, variance=optimal_hparams[10]))

    model = GPy.models.GPRegression(x_train.reshape(-1, 1), (y_train - y_mean).reshape(-1, 1), kernel=kernel,
                                    normalizer=False, noise_var=0.05)

    # We are done optimising, so we can now make predictions
    # for the entire dataset, including the test set.
    y_mean, y_std = model.predict(x_all.reshape(-1, 1))
    y_pred = draw_samples(y_mean, y_std)

    best_rmse = rms_error(y_all, y_pred)

    return optimal_hparams, best_rmse, y_mean, y_std


if __name__ == '__main__':
    args = parse_args()

    # Load the data.
    data = load_data_single_coordinate(lat=args.lat,
                                       lon=args.lon,
                                       num_months=args.num_months)
    x_train, y_train, x_test, y_test, x_valid, y_valid, x_all, y_all = data

    composite_filename = f"./logs/{args.model_type}_{args.acquisition_type}_{args.acquisition_optimiser_type}_" \
                         f"{str(args.lat).replace('.', 'p')}_{str(args.lon).replace('.', 'p')}_" \
                         f"{args.num_iterations}_{args.num_months}"

    # Here the flow of the program splits into two scenarios:
    # Run optimisation with domains found by grid search or
    # with default domains.
    if args.grid_search:
        num_hparams = 8 if args.kernel == 1 else 11  # The first kernel has 8 hparams, the second 11.
        grid_search_domains(k=args.k, kernel=args.kernel,
                            maximum_iterations=args.num_iterations, model_type=args.model_type,
                            initial_design_type=args.initial_design, acquisition_type=args.acquisition_type,
                            acquisition_weight=args.acquisition_weight,
                            acquisition_optimiser_type=args.acquisition_optimiser_type,
                            composite_filename=composite_filename)
    else:
        if args.kernel == 1:
            dom_tuples = [(0., 30.), (0., 30.), (0., 30.), (0., 30.),  # Length scale domains.
                          (0., 10.), (0., 10.), (0., 10.), (0., 10.)]  # Variance domains.
        else:
            dom_tuples = [(0., 10.), (0., 10.), (0., 10.), (0., 10.), (0., 10.),  # Length scale domains.
                          (0., 10.), (0., 10.), (0., 10.), (0., 10.), (0., 10.), (0., 10.)]  # Variance domains.

        tic = time.perf_counter()
        optimal_hparams, best_error, y_mean, y_std = optimise(maximum_iterations=args.num_iterations,
                                                              kernel_choice=args.kernel, dom_tuples=dom_tuples,
                                                              model_type=args.model_type,
                                                              initial_design_type=args.initial_design,
                                                              acquisition_type=args.acquisition_type,
                                                              acquisition_weight=args.acquisition_weight,
                                                              acquisition_optimiser_type=args.acquisition_optimiser_type
                                                              )
        toc = time.perf_counter()

        out = open(f"{composite_filename}.txt", "a+")
        log(out, args.kernel, args, optimal_hparams, best_error, dom_tuples, toc - tic)
        out.close()
        plot_model_predictions(x_train, x_valid, x_test, x_all,
                               y_train, y_valid, y_test, y_mean, y_std,
                          f"{composite_filename}.png")
