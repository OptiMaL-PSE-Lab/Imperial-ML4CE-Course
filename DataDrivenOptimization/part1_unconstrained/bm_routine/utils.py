# importing libraries
import matplotlib.pyplot as plt
import copy
import os
from datetime import datetime
from pylab import grid
import pickle
from tqdm import tqdm
import numpy as np
import copy
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# importing algorithms
from algorithms import *

# importing benchmarking routine
from bm_routine import *


def save_dict(target_folder, data_dict, timestamp):
    # Create a folder with current date and time
    folder_path = os.path.join(target_folder, timestamp)
    os.makedirs(folder_path, exist_ok=True)

    # Save dictionary to file within the created folder
    file_path = os.path.join(folder_path, "trajectories.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data_dict, f)


def ML4CE_uncon_eval(
    N_x_l,
    f_eval_l,
    functions_test,
    algorithms_test,
    reps,
    home_dir,
    SafeData=False,
):

    trajectories = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #############
    # Dimension #
    #############
    for i_run in range(len(N_x_l)):
        N_x_ = N_x_l[i_run]
        f_eval_ = f_eval_l[i_run]
        dim_S = "D" + str(N_x_)
        trajectories[dim_S] = {}
        bounds_ = np.array([[-7, 7] for i in range(N_x_)])

        #############
        # Functions #
        #############
        for i_function in functions_test:
            print("===================== ", i_function, "D" + str(N_x_))
            trajectories[dim_S][i_function] = {}
            trajectories[dim_S][i_function]["all means"] = {}
            trajectories[dim_S][i_function]["all 90"] = {}
            trajectories[dim_S][i_function]["all 10"] = {}
            trajectories[dim_S][i_function]["f_list"] = {}
            trajectories[dim_S][i_function]["x_list"] = {}
            trajectories[dim_S][i_function]["PID_traj"] = {}
            all_f_ = []
            randShift_l = np.random.uniform(-3, 3, (reps, N_x_))

            # save random shift
            trajectories[dim_S][i_function]["rand_shift"] = randShift_l

            ##############
            # Algorithms #
            ##############
            info = []

            for i_algorithm in algorithms_test:
                print("== ", str(i_algorithm.__name__))
                trajectories[dim_S][i_function][str(i_algorithm.__name__)] = []
                trajectories[dim_S][i_function]["f_list"][
                    str(i_algorithm.__name__)
                ] = []
                trajectories[dim_S][i_function]["x_list"][
                    str(i_algorithm.__name__)
                ] = []
                trajectories[dim_S][i_function]["PID_traj"][
                    str(i_algorithm.__name__)
                ] = []

                ###############
                # Repetitions #
                ###############
                for i_rep in tqdm(range(reps)):
                    # for i_rep in range(reps):
                    # random shift
                    x_shift_ = randShift_l[i_rep, :].reshape((N_x_, 1))
                    # test function
                    t_ = Test_function(i_function, N_x_, True, x_shift_, bounds_)
                    # algorithm
                    a, b, team_names, cids = i_algorithm(t_, N_x_, bounds_, f_eval_)
                    # post-processing
                    t_.best_f_list()  # List of best points so far
                    t_.pad_or_truncate(f_eval_)
                    # store result
                    trajectories[dim_S][i_function][str(i_algorithm.__name__)].append(
                        copy.deepcopy(t_.best_f_c)
                    )
                    trajectories[dim_S][i_function]["f_list"][
                        str(i_algorithm.__name__)
                    ].append(copy.deepcopy(t_.f_list))
                    trajectories[dim_S][i_function]["x_list"][
                        str(i_algorithm.__name__)
                    ].append(copy.deepcopy(t_.x_list))
                    # save trajectories for cstr_pid
                    if i_function == "cstr_pid_f":
                        trajectories[dim_S][i_function]["PID_traj"][
                            str(i_algorithm.__name__)
                        ].append(copy.deepcopy(t_.traj_pid))

                    # safe data in an overwriting fashion
                    if SafeData == True:
                        save_dict(home_dir, trajectories, timestamp)
                # statistics from each algorithm for a function
                l_ = np.array(
                    trajectories[dim_S][i_function][str(i_algorithm.__name__)]
                )
                m_ = np.mean(l_, axis=0)
                q10_ = np.quantile(l_, 0.10, axis=0)
                q90_ = np.quantile(l_, 0.90, axis=0)
                trajectories[dim_S][i_function]["all means"][
                    str(i_algorithm.__name__)
                ] = copy.deepcopy(m_)
                trajectories[dim_S][i_function]["all 90"][str(i_algorithm.__name__)] = (
                    copy.deepcopy(q10_)
                )
                trajectories[dim_S][i_function]["all 10"][str(i_algorithm.__name__)] = (
                    copy.deepcopy(q90_)
                )
                all_f_.append(copy.deepcopy(l_))
                info.append(
                    {
                        "alg_name": str(i_algorithm.__name__),
                        "team names": team_names,
                        "CIDs": cids,
                    }
                )
                # safe data in an overwriting fashion
                if SafeData == True:
                    save_dict(home_dir, trajectories, timestamp)
            # statistics for all algorithms for a function
            # trajectories[dim_S][i_function]['mean']   = np.mean(all_f_, axis=0)     # old
            # trajectories[dim_S][i_function]['median'] = np.median(all_f_, axis=0)   # old
            # trajectories[dim_S][i_function]['q 0']    = np.max(all_f_, axis=0)      # old
            # trajectories[dim_S][i_function]['q 100']  = np.min(all_f_, axis=0)      # old
            trajectories[dim_S][i_function]["mean"] = np.mean(all_f_, axis=(0, 1))
            trajectories[dim_S][i_function]["median"] = np.median(all_f_, axis=(0, 1))
            trajectories[dim_S][i_function]["q 0"] = np.max(all_f_, axis=(0, 1))
            trajectories[dim_S][i_function]["q 100"] = np.min(all_f_, axis=(0, 1))
            # safe data in an overwriting fashion
            if SafeData == True:
                save_dict(home_dir, trajectories, timestamp)

    # over-write one last time
    if SafeData == True:

        save_dict(home_dir, trajectories, timestamp)

        return info, trajectories, timestamp

    else:

        return info, trajectories, None


def ML4CE_uncon_table(
    trajectories,
    algs_test,
    funcs_test,
    multim,
    N_x_l,
    start_,
):
    """
    This function calculates the test results based on the trajectories and puts them in a format ready to be plotted in tables
    Furthermore, it normalizes the results among all algorithms for a given dimension and test function. By this, the height profile
    of the functions gets vanished, leading to an increased comparability among the performance on unimodal and multimodal functions.
    Reason: Unimodal tend to have higher function values in general which seems for the algorithms to perform worse

        multim: it gets the list of multi-modal functions, the rest are assumed to be uni-modal
        N_x_l: Input dimensions to be benchmarked
        start_: No of starting points per dimension
    """

    # computing table of results
    alg_perf = {}
    n_f = len(funcs_test)

    # for every algorithm
    for i_dim in range(len(N_x_l)):
        dim_S = "D" + str(N_x_l[i_dim])
        alg_perf[dim_S] = {}
        for i_alg in algs_test:
            print("==  ", str(i_alg.__name__), " ==")
            alg_perf[dim_S][str(i_alg.__name__)] = {}
            # for every function
            for i_fun in range(n_f):
                # retrive performance
                medall_ = trajectories[dim_S][funcs_test[i_fun]]["mean"]
                trial_ = trajectories[dim_S][funcs_test[i_fun]]["all means"][
                    str(i_alg.__name__)
                ]
                lowall_ = trajectories[dim_S][funcs_test[i_fun]]["q 100"]
                higall_ = trajectories[dim_S][funcs_test[i_fun]]["q 0"]
                # score performance
                perf_ = (higall_[start_[i_dim] :] - trial_[start_[i_dim] :]) / (
                    higall_[start_[i_dim] :] - lowall_[start_[i_dim] :]
                )

                alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i_fun]] = copy.deepcopy(
                    np.sum(perf_) / len(perf_)
                )

    cell_text_global = []

    for i_alg in algs_test:
        # Plot bars and create text labels for the table
        # for different dimensions
        for row in range(len(N_x_l)):
            dim_S = "D" + str(N_x_l[row])
            r_ = [
                alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i]] for i in range(n_f)
            ]
            cell_text_global.append(r_)

    # return array
    cell_text_global_array = np.array(cell_text_global)

    matrix_store_l = []

    # iteratively select all the performances for changing dimensions
    # (each column is a test function and each row is an algorihm)
    for i in range(len(N_x_l)):

        # select matrix for dimension i
        matrix = cell_text_global_array[i :: len(N_x_l)]
        # obtain column-wise the min-max values
        matrix_min = matrix.min(axis=0)
        matrix_max = matrix.max(axis=0)
        # normalize column-wise
        matrix_norm = (matrix - matrix_min) / (matrix_max - matrix_min)
        matrix_store_l.append(matrix_norm)

    ### This is super blocky and needs to be simplyfied ###

    # each matrix in here represents an input dimension. Each row is an algorithm, each column is a test function
    combined_matrix_norm = np.stack(matrix_store_l, axis=0)

    # have each matrix be an algorithm, each row be a dimension and each column a test function
    combined_matrix_norm_reshape = np.transpose(combined_matrix_norm, axes=(1, 0, 2))

    ### start building averages for the multimodal functions ###

    # Calculate row-wise averages of the first two original columns
    row_wise_averages_first = np.mean(
        combined_matrix_norm_reshape[:, :, : len(multim)], axis=2
    )

    # Expand dimensions to allow broadcasting
    row_wise_averages_first_expanded = row_wise_averages_first[:, :, np.newaxis]

    # Add the new column to each matrix
    result1 = np.concatenate(
        (combined_matrix_norm_reshape, row_wise_averages_first_expanded), axis=2
    )

    # Calculate row-wise averages of the second two original columns
    row_wise_averages_second = np.mean(
        combined_matrix_norm_reshape[:, :, len(multim) :], axis=2
    )

    # Expand dimensions to allow broadcasting
    row_wise_averages_second_expanded = row_wise_averages_second[:, :, np.newaxis]

    # Add the new column to each matrix
    result2 = np.concatenate((result1, row_wise_averages_second_expanded), axis=2)

    # Calculate row-wise averages of the two new columns
    row_wise_averages_new = np.mean(result2[:, :, :6], axis=2)

    # Expand dimensions to allow broadcasting
    row_wise_averages_new_expanded = row_wise_averages_new[:, :, np.newaxis]

    # Add the new column to each matrix
    result3 = np.concatenate((result2, row_wise_averages_new_expanded), axis=2)

    ### calculate the final row ###
    # Calculate column-wise averages across each matrix
    column_wise_averages = np.mean(result3, axis=1)

    # Stack the column-wise averages as an additional row to each matrix
    arr_with_avg_row = np.concatenate(
        (result3, column_wise_averages[:, np.newaxis, :]), axis=1
    )

    return np.around(arr_with_avg_row, decimals=2)


def ML4CE_uncon_table_plot(
    array, functions_test, algorithms_test, home_dir, timestamp, SafeFig=False
):

    # Select only the "All" row (last row) from the array for each algorithm
    all_row_idx = -1  # Index for "All" row
    cstr_pidf_idx = functions_test.index(
        "cstr_pid_f"
    )  # Index for the 'cstr_pidf' column

    # Create a matrix containing just the "All" row and 'cstr_pidf' column for each algorithm
    combined_matrix = []
    rows_names = []

    for i, algorithm in enumerate(algorithms_test):
        all_row = array[i][
            all_row_idx, cstr_pidf_idx
        ]  # Get the "All" row and 'cstr_pidf' column
        combined_matrix.append([all_row])  # Add as a single-column row
        rows_names.append(algorithm.__name__)  # Label the row with the algorithm name

    # Convert to NumPy array for consistency
    combined_matrix = np.array(combined_matrix)

    # Plot the table
    plt.figure(figsize=(8, 5))  # Adjust the size as needed
    plt.axis("off")

    # Display the table with just the algorithm names and 'cstr_pidf' values
    plt.table(
        cellText=combined_matrix,
        loc="center",
        cellLoc="center",
        colLabels=["cstr_pid_f"],  # Only one column
        rowLabels=rows_names,  # Algorithm names as row labels
    )

    plt.tight_layout()  # Adjust the layout to prevent overlapping

    if SafeFig:

        def directory_exists(directory_name):
            return os.path.isdir(directory_name)

        directory_name = os.path.join(home_dir, timestamp, "tables")

        if directory_exists(directory_name):
            plt.savefig(os.path.join(directory_name, "combined_results.png"))
        else:
            print(
                f"The directory '{directory_name}' does not exist. Creating directory."
            )
            os.mkdir(directory_name)
            plt.savefig(os.path.join(directory_name, "combined_results.png"))

        plt.close()

    else:
        plt.title("Combined Results for All Algorithms (cstr_pid_f)")
        plt.show()


def ML4CE_uncon_graph_abs(
    test_res, algs_test, funcs_test, N_x_l, home_dir, timestamp, SafeFig=False
):

    # Set the font properties globally
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.size": 28,
            "font.family": "lmr",
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
        }
    )

    # Use a different color map
    colors = plt.cm.tab10(
        np.linspace(0, 1, len(algs_test))
    )  # Change to 'viridis' color map
    # colors = [
    #     '#1A73B2',
    #     '#D62627',
    #     '#E476C2',
    #     '#0BBBCD',
    #     'grey',
    #     '#1A73B2',
    #     '#D62627',
    #     '#E476C2',
    #     '#0BBBCD',
    #     'grey',]
    line_styles = ["-", "--", "-.", ":"]
    alg_indices = {alg: i for i, alg in enumerate(algs_test)}

    n_f = len(funcs_test)

    for i_dim in range(len(N_x_l)):
        dim_S = "D" + str(N_x_l[i_dim])
        for i_fun in range(n_f):
            if funcs_test[i_fun] == "cstr_pid_f":
                fig, ax = plt.subplots(figsize=(18, 12))
            else:
                fig, ax = plt.subplots(figsize=(15, 15))
            # plt.figure(figsize=(15, 15))
            for i_alg in algs_test:
                trial_ = test_res[dim_S][funcs_test[i_fun]]["all means"][
                    str(i_alg.__name__)
                ]
                up_ = test_res[dim_S][funcs_test[i_fun]]["all 90"][str(i_alg.__name__)]
                down_ = test_res[dim_S][funcs_test[i_fun]]["all 10"][
                    str(i_alg.__name__)
                ]
                alg_index = alg_indices[i_alg]  # Get the index of the algorithm
                color = colors[alg_index]
                line_style = line_styles[alg_index % len(line_styles)]
                plt.plot(
                    trial_,
                    color=color,
                    linestyle=line_style,
                    lw=4,
                    label=str(i_alg.__name__),
                )
                x_ax = np.linspace(0, len(down_), len(down_), endpoint=False)
                plt.gca().fill_between(x_ax, down_, up_, color=color, alpha=0.2)

                # # Calculate the position of the vertical line based on the length of the trajectory
                # length = len(down_)
                # if length == 20:
                #     vline_pos = 5
                # elif length == 50:
                #     vline_pos = 10
                # elif length == 100:
                #     vline_pos = 15
                # else:
                #     vline_pos = None  # Or set a default value if needed

                # # Add the vertical line if a valid position is calculated
                # if vline_pos is not None:
                #     plt.axvline(x=vline_pos, color='black', linestyle='--', linewidth=2)

                # Determine the tick interval based on the length of down_
                if len(down_) < 100:
                    interval = 5
                else:
                    interval = 10

                # Setting x-axis ticks to integer values starting from 0 and showing every interval-th tick
                tick_positions = np.arange(0, len(down_), interval)
                if (
                    len(down_) - 1
                ) % interval != 0:  # If the last position is not already included
                    tick_positions = np.append(tick_positions, len(down_) - 1)
                tick_labels = np.arange(0, len(down_), interval)
                if len(tick_labels) < len(tick_positions):
                    tick_labels = np.append(tick_labels, len(down_) - 1)

                plt.xticks(tick_positions, tick_labels)

            legend_handles = []

            # if funcs_test[i_fun] == 'cstr_pid_f':
            #     legend_cust = [
            #     'SNOBFIT',
            #     'SRBF',
            #     'DYCORS',
            #     'SOP',
            #     'COBYLA',
            #     'TURBO',
            #     'COBYQA',
            #     'CUATRO',
            #     'CUATRO-pls',
            #     'BO',
            # ]
            # else:
            #     legend_cust = [
            #         'LSQM',
            #         'SNOBFIT',
            #         'SRBF',
            #         'DYCORS',
            #         'SOP',
            #         'COBYLA',
            #         'COBYQA',
            #         'CUATRO',
            #         'BO',
            #         'ENTMOOT',
            #     ]

            legend_cust = [alg.__name__ for alg in algs_test]

            for alg, label in zip(algs_test, legend_cust):
                alg_index = alg_indices[alg]
                color = colors[alg_index]
                line_style = line_styles[alg_index % len(line_styles)]
                handle = Line2D(
                    [0], [0], color=color, linestyle=line_style, lw=4, label=label
                )
                legend_handles.append(handle)

            plt.ylabel("Objective Function Value")
            plt.xlabel("Iterations")
            plt.yscale("log")
            if (
                funcs_test[i_fun] == "Rosenbrock_f"
                or funcs_test[i_fun] == "Levy_f"
                and dim_S == "D5"
            ):
                legend1 = plt.legend(
                    handles=legend_handles[:5], loc="lower left", frameon=False
                )
                legend2 = plt.legend(
                    handles=legend_handles[5:], loc="upper right", frameon=False
                )
                ax.add_artist(legend1)
                ax.add_artist(legend2)
            elif funcs_test[i_fun] == "cstr_pid_f":  # (0.735, 0.65)
                legend1 = plt.legend(
                    handles=legend_handles[:5], loc=(0.4642, 0.673), frameon=False
                )
                legend2 = plt.legend(
                    handles=legend_handles[5:], loc=(0.71, 0.673), frameon=False
                )
                ax.add_artist(legend1)
                ax.add_artist(legend2)
            else:

                plt.legend(handles=legend_handles, loc="best", frameon=False)
            plt.grid(which="major")
            plt.grid(which="minor", alpha=0.4)

            if SafeFig == True:

                def directory_exists(directory_name):
                    root_directory = os.getcwd()  # Root directory on Unix-like systems
                    directory_path = os.path.join(root_directory, directory_name)
                    return os.path.isdir(directory_path)

                directory_name = os.path.join(
                    home_dir, timestamp, "trajectory_plots_1D"
                )
                if directory_exists(directory_name):

                    plt.savefig(
                        directory_name
                        + "/{}_{}_1D.png".format(dim_S, funcs_test[i_fun]),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name
                        + "/{}_{}_1D.jpg".format(dim_S, funcs_test[i_fun]),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )
                else:
                    print(
                        f"The directory '{directory_name}' does not exist in the root directory."
                    )
                    os.mkdir(directory_name)
                    plt.savefig(
                        directory_name
                        + "/{}_{}_1D.png".format(dim_S, funcs_test[i_fun]),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name
                        + "/{}_{}_1D.jpg".format(dim_S, funcs_test[i_fun]),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )


def ML4CE_uncon_contour_allin1(
    functions_test,
    algorithms_test,
    N_x_,
    x_shift_origin,
    bounds_,
    bounds_plot,
    SafeFig=False,
):
    f_eval_ = 30  # trajectory length (= evaluation budget) --> Keep in mind that BO uses 10 datapoints to build the model when plotting
    track_x = True

    # Set the font properties globally
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.size": 28,
            "font.family": "lmr",
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
        }
    )

    line_styles = ["-", "--", "-.", ":"]

    # colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms_test)))
    # line_colors = ['k-','b-','g-','c-','r-']
    # colors = ['k','b','g','c','darkviolet']
    colors = ["#1A73B2", "#D62627", "#E476C2", "#0BBBCD", "grey"]
    # colors = ['black', 'white','darkslategray', 'gold', 'deepskyblue' ]
    alg_indices = {alg: i for i, alg in enumerate(algorithms_test)}
    fun_n = 0

    for fun_ in functions_test:
        f_contour = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)

        ####################################
        ### CONTOUR AND CONSTRAINT LINES ###
        ####################################
        # evaluate grid with vmap
        n_points = 200
        x1lb = bounds_plot[fun_n][0][0]
        x1ub = bounds_plot[fun_n][0][1]
        x1 = np.linspace(start=x1lb, stop=x1ub, num=n_points)
        x2lb = bounds_plot[fun_n][1][0]
        x2ub = bounds_plot[fun_n][1][1]
        x2 = np.linspace(start=x2lb, stop=x2ub, num=n_points)
        X1, X2 = np.meshgrid(x1, x2)

        # set function counter
        fun_n += 1

        # define plot
        plt.figure(figsize=(15, 15))
        ax3 = plt.subplot()

        # Initialize an empty array to store results
        y = np.empty((n_points, n_points))

        # Iterate over each element and apply the function
        # drawing contours
        print("Drawing Contours")
        for i in tqdm(range(n_points)):
            for j in range(n_points):
                y[i, j] = f_contour.fun_test(np.array([[X1[i, j], X2[i, j]]]))

        # change scale (all unconstrained functions look better like this)
        y = np.log2(y + 1)

        # Normalize the y values for the contour plot
        norm = mcolors.Normalize(vmin=np.min(y), vmax=np.max(y))

        # Plot the contour with normalization
        contour = plt.contourf(
            X1,
            X2,
            y,
            levels=50,
            cmap="Spectral_r",
            norm=norm,
            alpha=0.5,
            linewidths=0.5,
        )

        # set axis
        ax3.axis([x1lb, x1ub, x2lb, x2ub])

        ############################
        ### PERFORM OPTIMIZATION ###
        ############################
        for alg_ in algorithms_test:
            print(alg_)

            # color setting
            alg_index = alg_indices[alg_]
            color = colors[alg_index]

            # initiate test function
            f_plot = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)

            # perform optimization
            a, b, team_names, cids = alg_(f_plot, N_x_, bounds_, f_eval_, has_x0=True)

            # get an plot all evaluations
            # X_all = np.array(f_plot.x_list[:f_eval_+1])  # needs to be capped because of COBYLA not respecting the budget
            X_all = np.array(f_plot.x_list)
            ax3.scatter(X_all[:, 0], X_all[:, 1], marker="o", color=color, alpha=0.4)

            # re-test for feasibility (taken from constrained routine)
            f_feas = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)

            # all feasible since unconstrained
            X_feas = [
                x for x in X_all
            ]  # needs to be adapted later when merging both utils

            # produce function values for ranking
            for x in X_feas:
                f_feas.fun_test(x)

            # get best evaluations
            f_feas.best_f_list()

            # get best values
            best_values_x1 = [point[0] for point in f_feas.best_x]
            best_values_x2 = [point[1] for point in f_feas.best_x]

            # plot best values
            ax3.plot(
                best_values_x1,
                best_values_x2,
                colors[alg_index],
                linewidth=3,
                label=alg_.__name__,
                zorder=5,
            )

            # Highlight starting point
            ax3.plot(
                X_all[0, 0, 0],
                X_all[0, 1, 0],
                marker="X",
                color="black",
                markersize=15,
                zorder=6,
            )

            # highlight end point
            ax3.plot(
                best_values_x1[-1],
                best_values_x2[-1],
                colors[alg_index],
                marker="H",
                markersize=15,
                zorder=10,
            )

            # set lables
            plt.ylabel("$x_2$")
            plt.xlabel("$x_1$")

            # remove ticks
            plt.xticks([])
            plt.yticks([])

        legend_cust_dict = {
            "LS_QM_v2": "LSQM",
            "opt_SRBF": "SRBF",
            "opt_DYCORS": "DYCORS",
            "opt_SOP": "SOP",
            "COBYQA": "COBYQA",
            "opt_SnobFit": "SNOBFIT",
            "opt_COBYLA": "COBYLA",
            "opt_CUATRO": "CUATRO",
            "BO_np_scipy": "BO",
            "opt_ENTMOOT": "ENTMOOT",
        }

        legend_cust = [legend_cust_dict[key.__name__] for key in algorithms_test]

        # Create custom legend handles
        legend_handles = []
        for alg, label in zip(algorithms_test, legend_cust):
            alg_index = alg_indices[alg]
            color = colors[alg_index]
            handle = Line2D([0], [0], color=color, linestyle="-", lw=3, label=label)
            legend_handles.append(handle)

        # Add a custom legend handle for the constraint, start, and end
        # hline_handle = Line2D([0], [0], color='black', linestyle='--', lw=3, label='g(x) = 0', alpha=0.6)
        start_handle = Line2D(
            [0],
            [0],
            marker="X",
            color="black",
            linestyle="None",
            markersize=15,
            label="Start",
        )
        end_handle = Line2D(
            [0],
            [0],
            marker="H",
            color="black",
            linestyle="None",
            markersize=15,
            label="End",
        )
        # legend_handles.append(hline_handle)
        legend_handles.append(start_handle)
        legend_handles.append(end_handle)

        # Add legend with custom handles
        plt.legend(handles=legend_handles, loc="best")

        if SafeFig == True:

            def directory_exists(directory_name):
                root_directory = os.getcwd()  # Root directory on Unix-like systems
                directory_path = os.path.join(root_directory, directory_name)
                return os.path.isdir(directory_path)

            directory_name = "images/trajectory_plots_2D"
            if directory_exists(directory_name):

                if "LSQM" in legend_cust:
                    plt.savefig(
                        directory_name + "/{}_allin1_A.png".format(f_feas.func_type),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name + "/{}_allin1_A.jpg".format(f_feas.func_type),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )
                else:
                    plt.savefig(
                        directory_name + "/{}_allin1_B.png".format(f_feas.func_type),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name + "/{}_allin1_B.jpg".format(f_feas.func_type),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )
            else:
                print(
                    f"The directory '{directory_name}' does not exist in the root directory."
                )

            plt.close

        else:
            plt.show()


import matplotlib.colors as mcolors


def ML4CE_uncon_contour_allin1_smooth(
    functions_test,
    algorithms_test,
    N_x_,
    x_shift_origin,
    reps,
    bounds_,
    bounds_plot,
    SafeFig=False,
):

    # Set the font properties globally
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.size": 28,
            "font.family": "lmr",
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
        }
    )

    # data inputs, set to 60 for good visuals
    f_eval_ = 30  # trajectory length (= evaluation budget) --> Keep in mind that BO uses 10 datapoints to build the model when plotting

    track_x = True
    # line_colors = ['k-','b-','g-','c-'] looks like this is not needed
    # colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms_test)))
    """
    #CE7FA0 - Corallenrot
    #FAFCD7 - Orange
    #CDEAD1 - Gruen
    #9AC0DC - Himmelblau
    In combination with Greys
    colors = ['#CE7FA0', '#FAFCD7', '#CDEAD1', '#9AC0DC']
    """
    colors = ["#1A73B2", "#D62627", "#E476C2", "#0BBBCD", "grey"]
    # colors = ['k','b','g','c']

    alg_indices = {alg: i for i, alg in enumerate(algorithms_test)}
    fun_n = 0
    # perform plotting
    for fun_ in functions_test:
        f_contour = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)

        ####################################
        ### CONTOUR AND CONSTRAINT LINES ###
        ####################################
        # evaluate grid with vmap
        n_points = 100  # how 'round' the lines of the contour are
        x1lb = bounds_plot[fun_n][0][0]
        x1ub = bounds_plot[fun_n][0][1]
        x1 = np.linspace(start=x1lb, stop=x1ub, num=n_points)
        x2lb = bounds_plot[fun_n][1][0]
        x2ub = bounds_plot[fun_n][1][1]
        x2 = np.linspace(start=x2lb, stop=x2ub, num=n_points)
        X1, X2 = np.meshgrid(x1, x2)

        # set function counter
        fun_n += 1

        # define plot
        plt.figure(figsize=(15, 15))
        ax3 = plt.subplot()

        # Initialize an empty array to store results
        y = np.empty((n_points, n_points))

        # Iterate over each element and apply the function
        print("drawing contour")
        for i in tqdm(range(n_points)):
            for j in range(n_points):
                y[i, j] = f_contour.fun_test(np.array([[X1[i, j], X2[i, j]]]).flatten())

        # change scale (all unconstrained functions look better like this)
        y = np.log2(y + 1)

        # Normalize the y values for the contour plot
        norm = mcolors.Normalize(vmin=np.min(y), vmax=np.max(y))

        # Plot the contour with normalization
        contour = plt.contourf(
            X1,
            X2,
            y,
            levels=50,
            cmap="Spectral_r",
            norm=norm,
            alpha=0.5,
            linewidths=0.5,
        )

        # set axis
        ax3.axis([x1lb, x1ub, x2lb, x2ub])

        ############################
        ### PERFORM OPTIMIZATION ###
        ############################

        for alg_ in algorithms_test:

            alg_index = alg_indices[alg_]
            color = colors[alg_index]

            x_list = []
            for i_rep in range(reps):
                # initiate test function
                f_plot = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)
                # perform optimization
                a, b, team_names, cids = alg_(
                    f_plot, N_x_, bounds_, f_eval_, has_x0=True
                )
                # make sure the lenthgs of the lists match
                X_collect_pre = f_plot.x_list
                X_collect = []
                # make sure to even-out the trajectory lengths
                if len(X_collect_pre) < f_eval_:
                    x_last = X_collect_pre[-1]
                    X_collect = X_collect_pre + [x_last] * (
                        f_eval_ - len(X_collect_pre)
                    )
                elif len(X_collect_pre) > f_eval_:
                    X_collect = X_collect_pre[:f_eval_]
                else:
                    X_collect = X_collect_pre
                x_list.append(X_collect)

            # x_list and f_list now consist of all evaluations
            # now we build the mean
            X_all = np.array(x_list)
            X_all_m = np.mean(X_all, axis=0).reshape((f_eval_, 2))

            # re-test for feasibility (taken from constrained routine)
            f_feas = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)

            # all feasible since unconstrained
            X_feas = [
                x for x in X_all_m
            ]  # needs to be adapted later when merging both utils

            # produce function values for ranking
            for x in X_feas:
                f_feas.fun_test(x)

            # get best evaluations
            f_feas.best_f_list()

            # color setting
            alg_index = alg_indices[alg_]
            color = colors[alg_index]

            # # Find and plot unique evaluations (otherwise too crowded)
            # # another idea would be to adjust the alpha depending on how often a position is sampled
            # X_all_re = X_all.reshape((f_eval_*reps,2))
            # X_all_re_unique = np.unique(X_all_re, axis=0)
            # ax3.scatter(X_all_re_unique[:,0], X_all_re_unique[:,1], marker='o', color=color, alpha=0.5)

            # remove weird entries from CUATRO
            if alg_.__name__ == "opt_CUATRO":

                best_values_x1 = best_values_x1[9:]
                best_values_x2 = best_values_x2[9:]

            # plot
            best_values_x1 = [point[0] for point in f_feas.best_x]
            best_values_x2 = [point[1] for point in f_feas.best_x]

            # plot best values
            ax3.plot(
                best_values_x1,
                best_values_x2,
                colors[alg_index],
                linewidth=3,
                label=alg_.__name__,
                zorder=5,
            )

            # Highlight starting point
            ax3.plot(
                best_values_x1[0],
                best_values_x2[0],
                marker="X",
                color="black",
                markersize=15,
                zorder=6,
            )

            # highlight end point
            ax3.plot(
                best_values_x1[-1],
                best_values_x2[-1],
                colors[alg_index],
                marker="H",
                markersize=15,
                zorder=10,
            )

            # set lables
            plt.ylabel("$x_2$")
            plt.xlabel("$x_1$")

            # remove ticks
            plt.xticks([])
            plt.yticks([])

        legend_cust_dict = {
            "LS_QM_v2": "LSQM",
            "opt_SRBF": "SRBF",
            "opt_DYCORS": "DYCORS",
            "opt_SOP": "SOP",
            "COBYQA": "COBYQA",
            "opt_SnobFit": "SNOBFIT",
            "opt_COBYLA": "COBYLA",
            "opt_CUATRO": "CUATRO",
            "BO_np_scipy": "BO",
            "opt_ENTMOOT": "ENTMOOT",
        }

        legend_cust = [legend_cust_dict[key.__name__] for key in algorithms_test]

        # Create custom legend handles
        legend_handles = []
        for alg, label in zip(algorithms_test, legend_cust):
            alg_index = alg_indices[alg]
            color = colors[alg_index]
            handle = Line2D([0], [0], color=color, linestyle="-", lw=3, label=label)
            legend_handles.append(handle)

        # Add a custom legend handle for the constraint, start, and end
        start_handle = Line2D(
            [0],
            [0],
            marker="X",
            color="black",
            linestyle="None",
            markersize=15,
            label="Start",
        )
        end_handle = Line2D(
            [0],
            [0],
            marker="H",
            color="black",
            linestyle="None",
            markersize=15,
            label="End",
        )
        legend_handles.append(start_handle)
        legend_handles.append(end_handle)

        # Add legend with custom handles
        plt.legend(handles=legend_handles, loc="best")

        if SafeFig == True:

            def directory_exists(directory_name):
                root_directory = os.getcwd()  # Root directory on Unix-like systems
                directory_path = os.path.join(root_directory, directory_name)
                return os.path.isdir(directory_path)

            directory_name = "images/trajectory_plots_2D"
            if directory_exists(directory_name):

                if "LSQM" in legend_cust:
                    plt.savefig(
                        directory_name
                        + "/{}_allin1_A_smooth.png".format(f_feas.func_type),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name
                        + "/{}_allin1_A_smooth.jpg".format(f_feas.func_type),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )
                else:
                    plt.savefig(
                        directory_name
                        + "/{}_allin1_B_smooth.png".format(f_feas.func_type),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name
                        + "/{}_allin1_B_smooth.jpg".format(f_feas.func_type),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )
            else:
                print(
                    f"The directory '{directory_name}' does not exist in the root directory."
                )

            plt.close

        else:
            plt.show()


###################################################### UNUSED FUNCTIONS #############################################################
def ML4CE_uncon_graph_rel(test_res, algs_test, funcs_test, N_x_l):
    """
    relative performance
    """

    n_f = len(funcs_test)
    n_p = len(funcs_test) * len(N_x_l)
    nrow = int(n_p / 2) if n_p % 2 == 0 else int(n_p / 2) + 1

    for i_alg in algs_test:
        plt.figure(figsize=(16, 25))
        for i_dim in range(len(N_x_l)):
            dim_S = "D" + str(N_x_l[i_dim])
            for i_fun in range(n_f):
                plt.subplot(nrow, 2, n_f * i_dim + i_fun + 1)
                trial_ = test_res[dim_S][funcs_test[i_fun]]["all means"][
                    str(i_alg.__name__)
                ]
                up_ = test_res[dim_S][funcs_test[i_fun]]["all 90"][str(i_alg.__name__)]
                down_ = test_res[dim_S][funcs_test[i_fun]]["all 10"][
                    str(i_alg.__name__)
                ]
                medall_ = test_res[dim_S][funcs_test[i_fun]]["mean"]
                lowall_ = test_res[dim_S][funcs_test[i_fun]]["q 100"]
                higall_ = test_res[dim_S][funcs_test[i_fun]]["q 0"]
                plt.plot(
                    trial_, color="C" + str(i_fun), lw=3, label=str(i_alg.__name__)
                )
                plt.plot(medall_, "--", lw=2, label="median alg")
                plt.plot(lowall_, "--", lw=2, label="best alg")
                plt.plot(higall_, "--", lw=2, label="worst alg")
                x_ax = np.linspace(0, len(down_), len(down_), endpoint=True)
                plt.gca().fill_between(
                    x_ax, down_, up_, color="C" + str(i_fun), alpha=0.2
                )
                plt.ylabel("obj value")
                plt.xlabel("iterations")
                plt.legend(loc="best")
                plt.title(funcs_test[i_fun] + " " + dim_S + " convergence plot")
                grid(True)


def ML4CE_uncon_contours(
    obj_func,
    i_algorithm,
    bounds_plot,
    X_opt,
    xnew,
    samples_number,
    func_type,
    Cons=False,
    TR_plot=False,
    TR_l=False,
    PlotArrows=False,
    Zoom=False,
    SafeFig=False,
):
    """
    cons_plot_l -> list of functions for the constraints for plotting
    cons_eval_l -> list of functions for the constraints for evaluation
    bounds -> list of bounds as [[x1lb,x1ub],[x2lb,x2ub]]
    """
    # evaluate grid with vmap
    n_points = 100
    x1lb = bounds_plot[0][0]
    x1ub = bounds_plot[0][1]
    x1 = np.linspace(start=x1lb, stop=x1ub, num=n_points)
    x2lb = bounds_plot[1][0]
    x2ub = bounds_plot[1][1]
    x2 = np.linspace(start=x2lb, stop=x2ub, num=n_points)
    X1, X2 = np.meshgrid(x1, x2)

    # define plot
    plt.figure(figsize=(15, 15))
    ax3 = plt.subplot()

    # Initialize an empty array to store results
    y = np.empty((n_points, n_points))
    # Iterate over each element and apply the function
    for i in range(n_points):
        for j in range(n_points):
            y[i, j] = obj_func.fun_test(np.array([[X1[i, j], X2[i, j]]]))

    # add objective contour
    ax3.contour(X1, X2, y, 50)

    if Cons == True:

        # evaluate list of points for constraint plot
        con_list = [obj_func.con_plot(x_i) for x_i in x2]

        # add constraints to plot
        if func_type == "Rosenbrock_f":
            ax3.plot(
                con_list, x2, "black", linewidth=3
            )  #!!! careful here where to put x1 and x2
        else:
            ax3.plot(
                x1, con_list, "black", linewidth=3
            )  #!!! careful here where to put x1 and x2

    # add algorithm evaluations to plot
    # for CBO_TR we want to plot the initial samples to build the first GPs as *
    if i_algorithm.__name__ == "CBO_TR":
        # Calculate the length of the array
        array_length = len(X_opt[samples_number:, 0])
        # Define a color gradient from light red to dark red
        color_gradient = [(0, 0, 1 - i / array_length) for i in range(array_length)]
        # Plot each point with a color from the gradient
        for i in range(array_length):
            ax3.plot(
                X_opt[samples_number + i, 0],
                X_opt[samples_number + i, 1],
                marker="o",
                color=color_gradient[i],
            )
        # add initial samples to plot
        ax3.plot(X_opt[:samples_number, 0], X_opt[:samples_number, 1], "*")

    else:
        # Calculate the length of the array
        array_length = len(X_opt)
        # Define a color gradient from light red to dark red
        color_gradient = [(0, 0, 1 - i / array_length) for i in range(array_length)]
        # Plot each point with a color from the gradient
        for i in range(array_length):
            # ax3.plot(X_opt[i, 0], X_opt[i, 1], marker='o', color=color_gradient[i])
            ax3.plot(X_opt[i, 0], X_opt[i, 1], marker="o", color="grey")

    threshold = 2  # Define threshold here

    # connect best-so-far
    best_points = []
    best_value = float("inf")  # Initialize with a high value

    for point in X_opt:
        y = obj_func.fun_test(point)
        if y < best_value:
            best_value = y
            best_points.append(point)

    best_values_x1 = [point[0] for point in best_points]
    best_values_x2 = [point[1] for point in best_points]

    ax3.plot(best_values_x1, best_values_x2, marker="o", linestyle="-", color="black")

    if PlotArrows == True:

        # Extract x and y coordinates
        x_coords = X_opt[:, 0, 0]
        y_coords = X_opt[:, 1, 0]

        # Connect the points in order and add arrows
        for i in range(len(X_opt) - 1):
            plt.plot(
                [x_coords[i], x_coords[i + 1]],
                [y_coords[i], y_coords[i + 1]],
                color="blue",
            )
            # Calculate distance between consecutive points
            distance = np.sqrt(
                (x_coords[i + 1] - x_coords[i]) ** 2
                + (y_coords[i + 1] - y_coords[i]) ** 2
            )
            if distance > threshold:
                # Calculate midpoint
                midpoint = (
                    (x_coords[i] + x_coords[i + 1]) / 2,
                    (y_coords[i] + y_coords[i + 1]) / 2,
                )
                # Calculate direction
                direction = (
                    x_coords[i + 1] - x_coords[i],
                    y_coords[i + 1] - y_coords[i],
                )
                # Normalize direction
                direction /= np.linalg.norm(direction)
                # Add arrow
                plt.arrow(
                    midpoint[0],
                    midpoint[1],
                    direction[0],
                    direction[1],
                    head_width=0.2,
                    head_length=0.2,
                    fc="blue",
                    ec="blue",
                )

    # add trust regions to samples in plot
    if TR_plot == True:
        for i in range(X_opt[samples_number:, :].shape[0]):
            x_pos = X_opt[samples_number + i, 0]
            y_pos = X_opt[samples_number + i, 1]
            # plt.text(x_pos, y_pos, str(i))
            circle1 = plt.Circle(
                (x_pos, y_pos),
                radius=TR_l[i],
                color="black",
                fill=False,
                linestyle="--",
            )
            ax3.add_artist(circle1)

    # Add starting point to the trajectory
    ax3.plot(X_opt[0, 0, 0], X_opt[0, 1, 0], marker="s", color="black", markersize=10)

    # add final candidate to plot
    xnew = xnew.flatten()
    ax3.plot(xnew[0], xnew[1], marker="^", color="black", markersize=10)

    if Zoom == True:

        # Extract x and y coordinates
        x_coords = X_opt[:, 0, 0]
        y_coords = X_opt[:, 1, 0]

        # zoom-in
        x1lb_zoom = min(x_coords) - 1
        x1ub_zoom = max(x_coords) + 1
        x2lb_zoom = min(y_coords) - 1
        x2ub_zoom = max(y_coords) + 1

        ax3.axis([x1lb_zoom, x1ub_zoom, x2lb_zoom, x2ub_zoom])

    else:
        ax3.axis([x1lb, x1ub, x2lb, x2ub])

    if SafeFig == True:

        def directory_exists(directory_name):
            root_directory = os.getcwd()  # Root directory on Unix-like systems
            directory_path = os.path.join(root_directory, directory_name)
            return os.path.isdir(directory_path)

        directory_name = "images/trajectory_plots_2D"
        if directory_exists(directory_name):

            plt.savefig(
                directory_name
                + "/{}_{}.png".format(i_algorithm.__name__, obj_func.func_type)
            )
        else:
            print(
                f"The directory '{directory_name}' does not exist in the root directory."
            )

        plt.close

    else:
        plt.show()
