"""
This code has been adapted from: https://arxiv.org/pdf/2111.01732.pdf
"""

import time

import bayesnewton
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import objax
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def datetime_to_epoch(datetime):
    """
    Converts a datetime to a number
    args:
        datatime: is a pandas column
    """
    return datetime.astype("int64") // 1e9


def get_raw_data():
    raw_data = pd.read_csv("data/aq_data.csv")
    sites_df = pd.read_csv("data/laqn_sites.csv", sep=";")

    # filter sites not in london
    london_box = [[51.279, 51.684], [-0.533, 0.208]]  # lat  # lon

    sites_df = sites_df[
        (sites_df["Latitude"] > london_box[0][0])
        & (sites_df["Latitude"] < london_box[0][1])
    ]
    sites_df = sites_df[
        (sites_df["Longitude"] > london_box[1][0])
        & (sites_df["Longitude"] < london_box[1][1])
    ]

    raw_data = raw_data.merge(sites_df, left_on="site", right_on="SiteCode")
    raw_data = raw_data.dropna(subset=["pm25"])

    # convert to datetimes
    raw_data["date"] = pd.to_datetime(raw_data["date"])
    raw_data["epoch"] = datetime_to_epoch(raw_data["date"])

    # get data in date range
    data_range_start = "2019/02/18 00:00:00"
    data_range_end = "2019/02/25 23:59:59"

    raw_data = raw_data[
        (raw_data["date"] >= data_range_start)
        & (raw_data["date"] < data_range_end)
    ]

    X = np.array(raw_data[["epoch", "Longitude", "Latitude"]])
    Y = np.array(raw_data[["pm25"]])
    return X, Y, raw_data.copy()


def get_train_and_test_data(X, Y, raw_data_test):
    grid = True
    print(Y.shape)
    print("num data points =", Y.shape[0])

    xt = np.array(raw_data_test[["epoch", "Longitude", "Latitude"]])
    yt = np.array(raw_data_test[["pm25"]])

    t_test = xt[:, :1]
    R_test = xt[:, 1:]
    Y_test = yt[:, :]

    # the gridded approach:
    t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(
        X, Y
    )  # t is number of time steps, R coordinates, Y air pollution

    N_test = 20  # 50

    X1range = max(X[:, 1]) - min(X[:, 1])
    X2range = max(X[:, 2]) - min(X[:, 2])
    r1 = np.linspace(
        min(X[:, 1]) - 0.1 * X1range, max(X[:, 1]) + 0.1 * X1range, num=N_test
    )
    r2 = np.linspace(
        min(X[:, 2]) - 0.05 * X2range, max(X[:, 2]) + 0.05 * X2range, num=N_test
    )
    rA, rB = np.meshgrid(r1, r2)
    r = np.hstack(
        (rA.reshape(-1, 1), rB.reshape(-1, 1))
    )  # Flattening grid for use in kernel functions
    Rplot = np.tile(r, [t.shape[0], 1, 1])

    return t, R, Y, t_test, R_test, Y_test, Rplot, r1, r2


def plot_results(timesteps, X, Y, r1, r2, mu, R, z_opt):
    cmap = cm.viridis
    vmin = np.nanpercentile(Y, 1)
    vmax = np.nanpercentile(Y, 99)

    for time_step in range(timesteps.shape[0]):
        print(time_step)
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [20, 1]})
        f.set_figheight(8)

        im = a0.imshow(
            mu[time_step].T,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[r1[0], r1[-1], r2[0], r2[-1]],
            origin="lower",
        )

        a0.scatter(z_opt[:, 0], z_opt[:, 1], c="r", s=20, alpha=0.5)

        a0.set_xlim(r1[0], r1[-1])
        a0.set_ylim(r2[0], r2[-1])
        a0.set_xticks([], [])
        a0.set_yticks([], [])
        a0.set_title("pm25")

        a0.set_xlabel("Easting")
        a0.set_ylabel("Northing")
        a1.vlines(timesteps[time_step] / 24, -1, 1, "r")
        a1.set_xlabel("time (days)")
        a1.set_yticks([], [])
        a1.set_xlim(t[0] / 24, t[-1] / 24)

        f.savefig("output/output_%04d.pdf" % time_step)
        plt.close(f)


def get_kernel(X, Y):
    z1 = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), num=7)
    z2 = np.linspace(np.min(X[:, 2]), np.max(X[:, 2]), num=7)
    zA, zB = np.meshgrid(
        z1, z2
    )  # Adding additional dimension to inducing points grid
    z = np.hstack(
        (zA.reshape(-1, 1), zB.reshape(-1, 1))
    )  # Flattening grid for use in kernel functions

    kern_time = bayesnewton.kernels.Matern12(variance=1, lengthscale=5)
    kern_space0 = bayesnewton.kernels.Matern12(variance=1, lengthscale=1)
    kern_space1 = bayesnewton.kernels.Matern12(variance=1, lengthscale=1)
    kern_space = bayesnewton.kernels.Separable([kern_space0, kern_space1])

    return bayesnewton.kernels.SpatioTemporalKernel(
        temporal_kernel=kern_time,
        spatial_kernel=kern_space,
        z=z,  # initial spatial locations
        sparse=True,
        opt_z=True,
        conditional="Full",
    )


def main():
    X, Y, raw_data = get_raw_data()
    t, R, Y, t_test, R_test, Y_test, Rplot, r1, r2 = get_train_and_test_data(
        X, Y, raw_data
    )
    kernel = get_kernel(X, Y)

    likelihood = bayesnewton.likelihoods.Gaussian(variance=1)

    model = bayesnewton.models.MarkovVariationalGP(
        kernel=kernel, likelihood=likelihood, X=t, R=R, Y=Y
    )

    lr_adam = 0.05
    lr_newton = 0.5
    iters = 1
    opt_hypers = objax.optimizer.Adam(model.vars())
    energy = objax.GradValues(model.energy, model.vars())

    @objax.Function.with_vars(model.vars() + opt_hypers.vars())
    def train_op():
        model.inference(
            lr=lr_newton
        )  # perform inference and update variational params
        dE, E = energy()  # compute energy and its gradients w.r.t. hypers
        opt_hypers(lr_adam, dE)
        return E

    train_op = objax.Jit(train_op)

    t0 = time.time()
    for i in range(1, iters + 1):
        loss = train_op()
        print("iter %2d, energy: %1.4f" % (i, loss[0]))
    t1 = time.time()
    print("optimisation time: %2.2f secs" % (t1 - t0))

    print("calculating the posterior predictive distribution ...")
    t0 = time.time()
    posterior_mean, posterior_var = model.predict(X=t, R=Rplot)
    test_mean, test_var = model.predict(X=t_test, R=R_test)

    mae = mean_absolute_error(Y_test, test_mean)
    mse = mean_squared_error(Y_test, test_mean)
    rmse = mean_squared_error(Y_test, test_mean, squared=False)

    print(f"{mae=}")
    print(f"{mse=}")
    print(f"{rmse=}")

    nlpd = model.negative_log_predictive_density(X=t_test, R=R_test, Y=Y_test)
    t1 = time.time()
    print("prediction time: %2.2f secs" % (t1 - t0))
    print("nlpd: %2.3f" % nlpd)

    z_opt = model.kernel.z.value
    mu = bayesnewton.utils.transpose(posterior_mean.reshape(-1, 20, 20))

    plot_results(t, X, Y, r1, r2, mu, R, z_opt)


if __name__ == "__main__":
    main()
