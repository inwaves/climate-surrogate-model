import argparse
import numpy as np
from scipy.stats import multivariate_normal as mvn


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


def rms_error(y_true, y_pred) -> np.ndarray:
    """Calculate the root mean squared error."""
    # return np.sqrt(np.square(y_true - y_pred).mean())
    return np.sum(y_true - y_pred)


def parse_args():
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
    parser.add_argument("--k", default=2, type=int,
                        help="The number of domain values to explore per hyperparameter.")
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

    return parser.parse_args()
