import numpy as np
from ML4CE_WO_Wrapper import *
from datetime import datetime
from tqdm import tqdm
import statistics
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


##########################################
############ BENCHMARKING ################
##########################################
def ML4CE_con_eval(
    N_x_l,
    f_eval_l,
    functions_test,
    algorithms_test,
    reps,
    bounds=None,
    SafeData=False,
):

    trajectories = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save Info
    # lab_journal(home_dir, timestamp, file_name='INFO', SafeData=SafeData)

    #############
    # Dimension #
    #############
    for i_run in range(len(N_x_l)):
        N_x_ = N_x_l[i_run]
        f_eval_ = f_eval_l[i_run]
        dim_S = "D" + str(N_x_)
        trajectories[dim_S] = {}

        if bounds is None:
            bounds_ = np.array([[-5, 5] for i in range(N_x_)])
        else:
            bounds_ = bounds

        #############
        # Functions #
        #############
        for i_function in functions_test:
            print("===================== ", i_function, "D" + str(N_x_))
            trajectories[dim_S][i_function] = {}
            trajectories[dim_S][i_function]["all means"] = {}
            trajectories[dim_S][i_function]["all means g"] = {}
            trajectories[dim_S][i_function]["all 90"] = {}
            trajectories[dim_S][i_function]["all 10"] = {}
            trajectories[dim_S][i_function]["all 90 g"] = {}
            trajectories[dim_S][i_function]["all 10 g"] = {}
            trajectories[dim_S][i_function]["all means g1"] = {}
            trajectories[dim_S][i_function]["all 90 g1"] = {}
            trajectories[dim_S][i_function]["all 10 g1"] = {}
            trajectories[dim_S][i_function]["all means g2"] = {}
            trajectories[dim_S][i_function]["all 90 g2"] = {}
            trajectories[dim_S][i_function]["all 10 g2"] = {}
            trajectories[dim_S][i_function]["f_list"] = {}
            trajectories[dim_S][i_function]["x_list"] = {}
            trajectories[dim_S][i_function]["vio_g"] = {}
            trajectories[dim_S][i_function]["vio_g_values"] = {}
            trajectories[dim_S][i_function]["vio_g1_values"] = {}
            trajectories[dim_S][i_function]["vio_g2_values"] = {}
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
                trajectories[dim_S][i_function]["vio_g"][str(i_algorithm.__name__)] = []
                trajectories[dim_S][i_function]["vio_g_values"][
                    str(i_algorithm.__name__)
                ] = []
                trajectories[dim_S][i_function]["vio_g1_values"][
                    str(i_algorithm.__name__)
                ] = []
                trajectories[dim_S][i_function]["vio_g2_values"][
                    str(i_algorithm.__name__)
                ] = []
                ###############
                # Repetitions #
                ###############
                for i_rep in tqdm(range(reps)):
                    # random shift
                    x_shift_ = randShift_l[i_rep, :].reshape((N_x_, 1))
                    # test function
                    t_ = Test_function(i_function, N_x_, True)
                    # algorithm
                    team_names, cids = i_algorithm(
                        t_, bounds_, f_eval_, i_rep
                    )  # X_opt_plot, TR_l, xnew, backtrck_l, samples_number are for plotting
                    # post-processing
                    X_all_pre = t_.x_list
                    X_all = []
                    # make sure to even-out the trajectory lengths
                    if len(X_all_pre) < f_eval_:
                        x_last = X_all_pre[-1]
                        X_all = X_all_pre + [x_last] * (f_eval_ - len(X_all_pre))
                    elif len(X_all_pre) > f_eval_:
                        x_last = X_all_pre[:f_eval_][-1]
                        X_all = X_all_pre[:f_eval_] + [x_last] * (
                            f_eval_ - len(X_all_pre[:f_eval_])
                        )
                    else:
                        X_all = X_all_pre
                    X_all = np.array(X_all)
                    # capture constraint handling
                    if i_function != "WO_f":
                        trajectories[dim_S][i_function]["vio_g_values"][
                            str(i_algorithm.__name__)
                        ].append([-t_.con_test(x) for x in X_all])
                    # re-test for feasibility
                    f_feas = Test_function(i_function, N_x_, True)
                    if i_function == "WO_f":
                        X_feas = [
                            x
                            for x in X_all
                            if f_feas.WO_con1_test(x) > 0 and f_feas.WO_con2_test(x) > 0
                        ]
                        trajectories[dim_S][i_function]["vio_g1_values"][
                            str(i_algorithm.__name__)
                        ].append([-f_feas.WO_con1_test(x) for x in X_all])
                        trajectories[dim_S][i_function]["vio_g2_values"][
                            str(i_algorithm.__name__)
                        ].append([-f_feas.WO_con2_test(x) for x in X_all])
                    else:
                        X_feas = [x for x in X_all if f_feas.con_test(x) > 0]
                    # produce function values for ranking
                    for x in X_feas:
                        f_feas.fun_test(x)
                    # get best evaluations
                    f_feas.best_f_list()
                    f_feas.pad_or_truncate(f_eval_)
                    # store result
                    trajectories[dim_S][i_function][str(i_algorithm.__name__)].append(
                        copy.deepcopy(f_feas.best_f_c)
                    )
                    trajectories[dim_S][i_function]["f_list"][
                        str(i_algorithm.__name__)
                    ].append(copy.deepcopy(f_feas.f_list))
                    trajectories[dim_S][i_function]["x_list"][
                        str(i_algorithm.__name__)
                    ].append(copy.deepcopy(f_feas.x_list))
                    trajectories[dim_S][i_function]["vio_g"][
                        str(i_algorithm.__name__)
                    ].extend(copy.deepcopy(t_.g_list))
                    # # safe data in an overwriting fashion
                    # if SafeData == True: save_dict(home_dir, trajectories, timestamp)
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
                if i_function != "WO_f":
                    l_g_ = np.array(
                        trajectories[dim_S][i_function]["vio_g_values"][
                            str(i_algorithm.__name__)
                        ]
                    ).reshape(l_.shape)
                    m_g_ = np.mean(l_g_, axis=0)
                    q10_g_ = np.quantile(l_g_, 0.10, axis=0)
                    q90_g_ = np.quantile(l_g_, 0.90, axis=0)
                    trajectories[dim_S][i_function]["all means g"][
                        str(i_algorithm.__name__)
                    ] = copy.deepcopy(m_g_)
                    trajectories[dim_S][i_function]["all 90 g"][
                        str(i_algorithm.__name__)
                    ] = copy.deepcopy(q10_g_)
                    trajectories[dim_S][i_function]["all 10 g"][
                        str(i_algorithm.__name__)
                    ] = copy.deepcopy(q90_g_)
                else:
                    l_g1_ = np.array(
                        trajectories[dim_S][i_function]["vio_g1_values"][
                            str(i_algorithm.__name__)
                        ]
                    ).reshape(l_.shape)
                    l_g2_ = np.array(
                        trajectories[dim_S][i_function]["vio_g2_values"][
                            str(i_algorithm.__name__)
                        ]
                    ).reshape(l_.shape)
                    m_g1_ = np.mean(l_g1_, axis=0)
                    m_g2_ = np.mean(l_g2_, axis=0)
                    q10_g1_ = np.quantile(l_g1_, 0.10, axis=0)
                    q90_g1_ = np.quantile(l_g1_, 0.90, axis=0)
                    q10_g2_ = np.quantile(l_g2_, 0.10, axis=0)
                    q90_g2_ = np.quantile(l_g2_, 0.90, axis=0)
                    trajectories[dim_S][i_function]["all means g1"][
                        str(i_algorithm.__name__)
                    ] = copy.deepcopy(m_g1_)
                    trajectories[dim_S][i_function]["all 90 g1"][
                        str(i_algorithm.__name__)
                    ] = copy.deepcopy(q10_g1_)
                    trajectories[dim_S][i_function]["all 10 g1"][
                        str(i_algorithm.__name__)
                    ] = copy.deepcopy(q90_g1_)
                    trajectories[dim_S][i_function]["all means g2"][
                        str(i_algorithm.__name__)
                    ] = copy.deepcopy(m_g2_)
                    trajectories[dim_S][i_function]["all 90 g2"][
                        str(i_algorithm.__name__)
                    ] = copy.deepcopy(q10_g2_)
                    trajectories[dim_S][i_function]["all 10 g2"][
                        str(i_algorithm.__name__)
                    ] = copy.deepcopy(q90_g2_)
                all_f_.append(copy.deepcopy(l_))
                info.append(
                    {
                        "alg_name": str(i_algorithm.__name__),
                        "team names": team_names,
                        "CIDs": cids,
                    }
                )
                # # safe data in an overwriting fashion
                # if SafeData == True: save_dict(home_dir, trajectories, timestamp)
            # statistics for all algorithms for a function
            trajectories[dim_S][i_function]["mean"] = np.mean(all_f_, axis=(0, 1))
            trajectories[dim_S][i_function]["median"] = np.median(all_f_, axis=(0, 1))
            trajectories[dim_S][i_function]["q 0"] = np.max(all_f_, axis=(0, 1))
            trajectories[dim_S][i_function]["q 100"] = np.min(all_f_, axis=(0, 1))
            # safe data in an overwriting fashion
            # if SafeData == True: save_dict(home_dir, trajectories, timestamp)

    # over-write one last time
    if SafeData == True:

        # Safe Data
        # save_dict(home_dir, trajectories, timestamp)

        return info, trajectories

    else:

        return info, trajectories


def ML4CE_con_table_coursework(
    trajectories,
    algs_test,
    funcs_test,
    N_x_l,
    Normalize=False,
):

    # we take into account the entire trajectory length
    start_ = 0

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

    ### DIMENSION ###
    for i_dim in range(len(N_x_l)):
        dim_S = "D" + str(N_x_l[i_dim])
        alg_perf[dim_S] = {}

        ### ALGORITHM ###
        for i_alg in algs_test:
            print("==  ", str(i_alg.__name__), " ==")
            alg_perf[dim_S][str(i_alg.__name__)] = {}

            ### FUNCTION ###
            for i_fun in range(n_f):
                # retrive performance
                medall_ = trajectories[dim_S][funcs_test[i_fun]]["mean"]
                trial_ = trajectories[dim_S][funcs_test[i_fun]]["all means"][
                    str(i_alg.__name__)
                ]
                lowall_ = trajectories[dim_S][funcs_test[i_fun]]["q 100"]
                higall_ = trajectories[dim_S][funcs_test[i_fun]]["q 0"]
                # score performance
                # since we're providing starting points to the algorithms we can use the entire trajectory
                # instead of cutting the inital evaluations out as we do in the unconstrained benchmarking
                # perf_ = ( (higall_[:] - trial_[:])
                #         /(higall_[:] - lowall_[:]) )
                # perf_ = ( (higall_[start_[i_dim]:] - trial_[start_[i_dim]:])
                #         /(higall_[start_[i_dim]:] - lowall_[start_[i_dim]:]) )
                perf_ = (higall_[start_:] - trial_[start_:]) / (
                    higall_[start_:] - lowall_[start_:]
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

    # Calculate row-wise averages of all columns
    row_wise_averages_first = np.mean(combined_matrix_norm_reshape, axis=2)

    # Expand dimensions to allow broadcasting
    row_wise_averages_first_expanded = row_wise_averages_first[:, :, np.newaxis]

    # Add the new column to each matrix
    result1 = np.concatenate(
        (combined_matrix_norm_reshape, row_wise_averages_first_expanded), axis=2
    )

    ### calculate the final row ###
    # Calculate column-wise averages across each matrix
    column_wise_averages = np.mean(result1, axis=1)

    # Stack the column-wise averages as an additional row to each matrix
    arr_with_avg_row = np.concatenate(
        (result1, column_wise_averages[:, np.newaxis, :]), axis=1
    )

    # pure test results # shape (num_alg, num_dim + 1, num_fun + 1)
    test_res = np.around(arr_with_avg_row, decimals=2)

    ### Now attach the constraint violation ###
    # computing table of results
    n_f = len(funcs_test)
    vio_dict = {}

    ### DIMENSION ###
    for i_dim in range(len(N_x_l)):
        dim_S = "D" + str(N_x_l[i_dim])

        vio_dict[dim_S] = {}

        ### ALGORITHM ###
        for i_alg in algs_test:

            vio_dict[dim_S][str(i_alg.__name__)] = {}

            ### FUNCTION ###
            for i_fun in range(n_f):

                vio_list = trajectories[dim_S][funcs_test[i_fun]]["vio_g"][
                    str(i_alg.__name__)
                ]
                vio_dict[dim_S][str(i_alg.__name__)][funcs_test[i_fun]] = {}
                vio_dict[dim_S][str(i_alg.__name__)][funcs_test[i_fun]]["vio_g"] = {}

                # mean violation
                valid_values = [value for value in vio_list if value is not None]

                if valid_values:
                    mean_vio = statistics.mean(valid_values)
                else:
                    mean_vio = 0

                vio_dict[dim_S][str(i_alg.__name__)][funcs_test[i_fun]]["vio_g"][
                    "mean_vio"
                ] = mean_vio

                # mean no of constrain satisfaction
                vio_dict[dim_S][str(i_alg.__name__)][funcs_test[i_fun]]["vio_g"][
                    "perc_nonvio"
                ] = (vio_list.count(None) / len(vio_list)) * 100

    return cell_text_global_array, test_res, vio_dict


def ML4CE_con_table_plot_coursework(info, test_res_raw, test_res_norm, vio_dict):

    columns = [
        "Score",
        "Rank",
        "Feasible Samples [%]",
    ]  # !!! Do not change this order !!!
    rows_names_l = [str(list(info[i[0]].values())) for i in enumerate(info)]

    ranks = test_res_norm[:, 0, 0].reshape(test_res_raw.shape)
    mean_vio = []
    for alg in vio_dict["D2"]:
        mean_vio.append(vio_dict["D2"][alg]["WO_f"]["vio_g"]["perc_nonvio"])
    perc_nonvio = np.array(mean_vio).reshape(ranks.shape)
    array = np.concatenate(
        (np.round(test_res_raw, 2), ranks, np.round(perc_nonvio, 2)), axis=1
    )

    plt.figure()
    plt.axis("off")
    plt.table(
        cellText=array,
        loc="center",
        cellLoc="center",
        colLabels=columns,
        rowLabels=rows_names_l,
        # fontsize=10,
    )


def ML4CE_con_graph_abs(test_res, algs_test, funcs_test, N_x_l, SafeFig=False):

    # # Set the font properties globally
    # plt.rcParams.update({
    #     'text.usetex': True,
    #     'font.size': 28,
    #     'font.family': 'lmr',
    #     'xtick.labelsize': 26,
    #     'ytick.labelsize': 26,
    # })
    # colors = plt.cm.tab10(np.linspace(0, 1, len(algs_test)))
    colors = ["#1A73B2", "#D62627", "#E476C2", "#0BBBCD"]
    line_styles = ["-", "--", "-.", ":"]
    alg_indices = {alg: i for i, alg in enumerate(algs_test)}

    n_f = len(funcs_test)

    for i_dim in range(len(N_x_l)):
        dim_S = "D" + str(N_x_l[i_dim])
        for i_fun in range(n_f):
            plt.figure()
            for i_alg in algs_test:
                trial_ = test_res[dim_S][funcs_test[i_fun]]["all means"][
                    str(i_alg.__name__)
                ]
                up_ = test_res[dim_S][funcs_test[i_fun]]["all 90"][str(i_alg.__name__)]
                down_ = test_res[dim_S][funcs_test[i_fun]]["all 10"][
                    str(i_alg.__name__)
                ]
                alg_index = alg_indices[i_alg]
                color = colors[alg_index]
                line_style = line_styles[alg_index % len(line_styles)]
                plt.plot(
                    trial_,
                    color=color,
                    linestyle=line_style,
                    lw=4,
                    label=str(i_alg.__name__),
                    markersize=10,
                )
                x_ax = np.linspace(0, len(down_), len(down_), endpoint=False)
                plt.fill_between(x_ax, down_, up_, color=color, alpha=0.2)

                # # Calculate the position of the vertical line based on the length of the trajectory
                # length = len(down_)
                # if length == 20:
                #     vline_pos = 10
                # elif length == 50:
                #     vline_pos = 10
                # elif length == 100:
                #     vline_pos = 15
                # else:
                # vline_pos = None

                # # Add the vertical line if a valid position is calculated
                # if vline_pos is not None:
                #     plt.axvline(x=vline_pos, color='black', linestyle='--', linewidth=2)

                # Setting x-axis ticks to integer values starting from 0 and showing every 5th tick
                tick_positions = np.arange(0, len(down_), 5)
                if (
                    len(down_) - 1
                ) % 5 != 0:  # If the last position is not already included
                    tick_positions = np.append(tick_positions, len(down_) - 1)
                tick_labels = np.arange(0, len(down_), 5)
                if len(tick_labels) < len(tick_positions):
                    tick_labels = np.append(tick_labels, len(down_) - 1)

                plt.xticks(tick_positions, tick_labels)

            legend_handles = []
            legend_cust = algs_test
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
            if funcs_test[i_fun] != "WO_f":
                plt.yscale("log")
            plt.legend(handles=legend_handles, loc="best", frameon=False)
            plt.grid(True)

            # if SafeFig == True:

            #         def directory_exists(directory_name):
            #             root_directory = os.getcwd()  # Root directory on Unix-like systems
            #             directory_path = os.path.join(root_directory, directory_name)
            #             return os.path.isdir(directory_path)

            #         directory_name = os.path.join(home_dir, timestamp, 'trajectory_plots_1D')
            #         if directory_exists(directory_name):

            #             plt.savefig(directory_name + '/{}_{}_1D.png'.format(dim_S, funcs_test[i_fun]), bbox_inches = 'tight')
            #             plt.savefig(directory_name + '/{}_{}_1D.jpg'.format(dim_S, funcs_test[i_fun]), format = 'jpg', bbox_inches = 'tight', dpi=300)
            #         else:
            #             print(f"The directory '{directory_name}' does not exist in the root directory.")
            #             os.mkdir(directory_name)
            #             plt.savefig(directory_name + '/{}_{}_1D.png'.format(dim_S, funcs_test[i_fun]), bbox_inches = 'tight')
            #             plt.savefig(directory_name + '/{}_{}_1D.jpg'.format(dim_S, funcs_test[i_fun]), format = 'jpg', bbox_inches = 'tight', dpi=300)


def ML4CE_con_graph_abs_g1_coursework(
    test_res, algs_test, funcs_test, N_x_l, SafeFig=False
):

    # colors = plt.cm.tab10(np.linspace(0, 1, len(algs_test)))
    colors = ["#1A73B2", "#D62627", "#E476C2", "#0BBBCD"]
    # colors = plt.cm.tab10(np.linspace(0, 1, len(algs_test)))
    line_styles = ["-", "--", "-.", ":"]
    alg_indices = {alg: i for i, alg in enumerate(algs_test)}

    n_f = len(funcs_test)

    for i_dim in range(len(N_x_l)):
        dim_S = "D" + str(N_x_l[i_dim])
        for i_fun in range(n_f):
            plt.figure()
            for i_alg in algs_test:
                trial_ = test_res[dim_S][funcs_test[i_fun]]["all means g1"][
                    str(i_alg.__name__)
                ]
                up_ = test_res[dim_S][funcs_test[i_fun]]["all 90 g1"][
                    str(i_alg.__name__)
                ]
                down_ = test_res[dim_S][funcs_test[i_fun]]["all 10 g1"][
                    str(i_alg.__name__)
                ]
                alg_index = alg_indices[i_alg]
                color = colors[alg_index]
                line_style = line_styles[alg_index % len(line_styles)]
                plt.plot(
                    trial_,
                    color=color,
                    linestyle=line_style,
                    lw=4,
                    label=str(i_alg.__name__),
                    markersize=10,
                )
                x_ax = np.linspace(0, len(down_), len(down_), endpoint=False)
                plt.fill_between(x_ax, down_, up_, color=color, alpha=0.2)

                # Add horizontal line at y=0
                plt.axhline(y=0, color="black", linestyle="--", linewidth=3)

                # Setting x-axis ticks to integer values starting from 0 and showing every 5th tick
                tick_positions = np.arange(0, len(down_), 5)
                if (
                    len(down_) - 1
                ) % 5 != 0:  # If the last position is not already included
                    tick_positions = np.append(tick_positions, len(down_) - 1)
                tick_labels = np.arange(0, len(down_), 5)
                if len(tick_labels) < len(tick_positions):
                    tick_labels = np.append(tick_labels, len(down_) - 1)

                plt.xticks(tick_positions, tick_labels)

            legend_handles = []
            legend_cust = algs_test
            for alg, label in zip(algs_test, legend_cust):
                alg_index = alg_indices[alg]
                color = colors[alg_index]
                line_style = line_styles[alg_index % len(line_styles)]
                handle = Line2D(
                    [0], [0], color=color, linestyle=line_style, lw=4, label=label
                )
                legend_handles.append(handle)

            # Add a custom legend handle for the horizontal line at y=0
            hline_handle = Line2D(
                [0], [0], color="black", linestyle="--", lw=3, label="$g_1(x) = 0$"
            )
            legend_handles.append(hline_handle)

            plt.ylabel("Constraint Function Value")
            plt.xlabel("Iterations")

            if funcs_test[i_fun] == "Rosenbrock_f":
                top = 2
            if funcs_test[i_fun] == "Antonio_f":
                top = 2.5
            if funcs_test[i_fun] == "Matyas_f":
                top = 10
            # plt.ylim(top=top)
            plt.legend(handles=legend_handles, frameon=False)
            plt.grid(True)

            # if SafeFig == True:

            #     def directory_exists(directory_name):
            #         root_directory = os.getcwd()  # Root directory on Unix-like systems
            #         directory_path = os.path.join(root_directory, directory_name)
            #         return os.path.isdir(directory_path)

            #     directory_name = os.path.join(home_dir, timestamp, 'trajectory_plots_1D')
            #     if directory_exists(directory_name):

            #         plt.savefig(directory_name + '/{}_{}_g_1D.png'.format(dim_S, funcs_test[i_fun]), bbox_inches = 'tight')
            #         plt.savefig(directory_name + '/{}_{}_g_1D.jpg'.format(dim_S, funcs_test[i_fun]), format = 'jpg', bbox_inches = 'tight', dpi=300)
            #     else:
            #         print(f"The directory '{directory_name}' does not exist in the root directory.")
            #         os.mkdir(directory_name)
            #         plt.savefig(directory_name + '/{}_{}_g_1D.png'.format(dim_S, funcs_test[i_fun]), bbox_inches = 'tight')
            #         plt.savefig(directory_name + '/{}_{}_g_1D.jpg'.format(dim_S, funcs_test[i_fun]), format = 'jpg', bbox_inches = 'tight', dpi=300)


def ML4CE_con_graph_abs_g2_coursework(
    test_res, algs_test, funcs_test, N_x_l, SafeFig=False
):

    # colors = plt.cm.tab10(np.linspace(0, 1, len(algs_test)))
    colors = ["#1A73B2", "#D62627", "#E476C2", "#0BBBCD"]
    # colors = plt.cm.tab10(np.linspace(0, 1, len(algs_test)))
    line_styles = ["-", "--", "-.", ":"]
    alg_indices = {alg: i for i, alg in enumerate(algs_test)}

    n_f = len(funcs_test)

    for i_dim in range(len(N_x_l)):
        dim_S = "D" + str(N_x_l[i_dim])
        for i_fun in range(n_f):
            plt.figure()
            for i_alg in algs_test:
                trial_ = test_res[dim_S][funcs_test[i_fun]]["all means g2"][
                    str(i_alg.__name__)
                ]
                up_ = test_res[dim_S][funcs_test[i_fun]]["all 90 g2"][
                    str(i_alg.__name__)
                ]
                down_ = test_res[dim_S][funcs_test[i_fun]]["all 10 g2"][
                    str(i_alg.__name__)
                ]
                alg_index = alg_indices[i_alg]
                color = colors[alg_index]
                line_style = line_styles[alg_index % len(line_styles)]
                plt.plot(
                    trial_,
                    color=color,
                    linestyle=line_style,
                    lw=4,
                    label=str(i_alg.__name__),
                    markersize=10,
                )
                x_ax = np.linspace(0, len(down_), len(down_), endpoint=False)
                plt.fill_between(x_ax, down_, up_, color=color, alpha=0.2)

                # Add horizontal line at y=0
                plt.axhline(y=0, color="black", linestyle="--", linewidth=3)

                # Setting x-axis ticks to integer values starting from 0 and showing every 5th tick
                tick_positions = np.arange(0, len(down_), 5)
                if (
                    len(down_) - 1
                ) % 5 != 0:  # If the last position is not already included
                    tick_positions = np.append(tick_positions, len(down_) - 1)
                tick_labels = np.arange(0, len(down_), 5)
                if len(tick_labels) < len(tick_positions):
                    tick_labels = np.append(tick_labels, len(down_) - 1)

                plt.xticks(tick_positions, tick_labels)

            legend_handles = []
            legend_cust = algs_test
            for alg, label in zip(algs_test, legend_cust):
                alg_index = alg_indices[alg]
                color = colors[alg_index]
                line_style = line_styles[alg_index % len(line_styles)]
                handle = Line2D(
                    [0], [0], color=color, linestyle=line_style, lw=4, label=label
                )
                legend_handles.append(handle)

            # Add a custom legend handle for the horizontal line at y=0
            hline_handle = Line2D(
                [0], [0], color="black", linestyle="--", lw=3, label="$g_2(x) = 0$"
            )
            legend_handles.append(hline_handle)

            plt.ylabel("Constraint Function Value")
            plt.xlabel("Iterations")

            if funcs_test[i_fun] == "Rosenbrock_f":
                top = 2
            if funcs_test[i_fun] == "Antonio_f":
                top = 2.5
            if funcs_test[i_fun] == "Matyas_f":
                top = 10
            # plt.ylim(top=top)
            plt.legend(handles=legend_handles, frameon=False)
            plt.grid(True)
