import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn


def draw_samples(y_mean, y_std):
    """Make some predictions by drawing from GP distributions."""
    mean = y_mean.reshape(-1)
    cov = np.diag(y_std.reshape(-1))
    y_pred = mvn.rvs(mean, cov, 1)

    return y_pred


def log(file_out, kernel, exp_args, hparams, final_error, domains, time_elapsed):
    """Log the results of the optimisation."""
    file_out.writelines(f"----------------------------------------------------\n"
                        f"Experiment took: {time_elapsed:.2f} seconds\n")
    file_out.writelines(f"Experiment parameters:\n"
                        f"Latitude: {exp_args.lat}, Longitude: {exp_args.lon}\n"
                        f"Number of months: {exp_args.num_months}, Number of iterations: {exp_args.num_iterations}\n"
                        f"Domain of lengthscale1: {domains[0]}, lengthscale2: {domains[1]}, "
                        f"lengthscale3: {domains[2]} \n "
                        f"Domain of variance1: {domains[3]}, variance2: {domains[4]}\n")

    if kernel == 1:
        file_out.writelines(f"Experiment results:\n"
                            f"ls1: {str(hparams[0])}, ls2: {str(hparams[1])}, "
                            f"ls3: {str(hparams[2])}, ls4 = {str(hparams[3])}"
                            f"v1: {str(hparams[4])}, v2: {str(hparams[5])}"
                            f"v3: {str(hparams[6])}, v4: {str(hparams[7])}"
                            f", best_error: {final_error}\n")
    else:
        file_out.writelines(f"Experiment results:\n"
                            f"ls1: {str(hparams[0])}, ls2: {str(hparams[1])}, "
                            f"ls3: {str(hparams[2])}, ls4 = {str(hparams[3])}, ls5 = {str(hparams[4])}"
                            f"v1: {str(hparams[5])}, v2: {str(hparams[6])}, v3: {str(hparams[7])}"
                            f"v4: {str(hparams[8])}, v5: {str(hparams[9])}, v6: {str(hparams[10])}"
                            f", best_error: {final_error}\n")


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
    parser.add_argument("--grid_search", default=False, type=bool,
                        help="Whether to perform a grid search over the domains of the hyperparameters. ")
    parser.add_argument("--initial_design", default="random", type=str,
                        help="The type of initial design, where to collect points. "
                             "Defaults to random. Choose from random, latin.")
    parser.add_argument("--k", default=2, type=int,
                        help="The number of domain values to explore per hyperparameter.")
    parser.add_argument("--kernel", default=1, type=int,
                        help="Which kernel to use. 1 is the RBF-Cosine, 2 is the RBF-Cosine-Noise.")
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


def load_data(num_months: int = 1980, training_proportion: float = 0.8) -> tuple:
    pass


def load_data_single_coordinate(lat: float = 51.875, lon: float = 0.9375, num_months: int = 1980,
                                training_proportion: float = 0.8) -> tuple:
    """Load data from a .csv file with the specified parameters.
        :param lat: The latitude of the location.
        :param lon: The longitude of the location.
        :param num_months: The total number of months to consider starting in Jan 1850.
        :param training_proportion: The proportion of the data to use for training.
        :return: A tuple containing the training set, the test set, as well as the two sets combined.
    """
    # TODO: load global dataset instead of one single point. x[t] = [lat, lon, t]
    lat_str, lon_str = str(lat).replace(".", "_"), str(lon).replace(".", "_")
    pr_df = pd.read_csv(f"./data/{lat_str}x{lon_str}.csv")

    last_train_index = int(num_months * training_proportion)
    last_validation_index = int(last_train_index + ((num_months - last_train_index) // 3))

    training_set = pr_df.iloc[0:last_train_index]
    validation_set = pr_df.iloc[last_train_index:last_validation_index]
    test_set = pr_df.iloc[last_validation_index:num_months]

    x_train = np.linspace(1, last_train_index,
                          last_train_index, dtype=np.int64)
    x_valid = np.linspace(last_train_index, last_validation_index,
                          last_validation_index - last_train_index, dtype=np.int64)
    x_test = np.linspace(last_validation_index, num_months,
                         num_months - last_validation_index, dtype=np.int64)
    x_all = np.concatenate((x_train, x_valid, x_test))

    # Renormalise precipitation measurements since they're all approximately r*10^-5
    y_train = (training_set["pr"].to_numpy()) * 10 ** 5
    y_valid = (validation_set["pr"].to_numpy()) * 10 ** 5
    y_test = test_set["pr"].to_numpy() * 10 ** 5
    y_all = np.concatenate([y_train, y_valid, y_test])

    return x_train, y_train, x_valid, y_valid, x_test, y_test, x_all, y_all


def plot_model_predictions(x_train: np.ndarray, x_valid: np.ndarray, x_test: np.ndarray, x_all: np.ndarray,
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
    plt.ylabel("residual")
    plt.scatter(x_train, y_train - y_train.mean(), lw=1, color="b", label="training dataset")
    plt.scatter(x_valid, y_valid - y_valid.mean(), lw=1, color="y", label="validation dataset")
    plt.scatter(x_test, y_test - y_test.mean(), lw=1, color="r", label="testing dataset")
    plt.plot(x_all, y_mean, lw=3, color="g", label="GP mean")
    plt.fill_between(x_all, (y_mean + y_std).reshape(y_mean.shape[0]), (y_mean - y_std).reshape(y_mean.shape[0]),
                     facecolor="b", alpha=0.3, label="confidence")
    plt.legend(loc="upper left")
    plt.savefig(filename)
