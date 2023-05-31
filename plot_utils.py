import collections
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import scipy
import seaborn as sns


def print_accuracies(
    valid_window_sizes, dynamic_acc, dynamic_se, ws_acc, ws_se, max_acc, max_se
):
    log_window_sizes = ["Adaptive", "Max"] + [str(i) for i in valid_window_sizes]
    log_accs = [dynamic_acc, max_acc] + ws_acc
    log_ses = [dynamic_se, max_se] + ws_se

    print(f"Window Size: Accuracy")
    for ws, acc, se in zip(log_window_sizes, log_accs, log_ses):
        print(f"{ws:<11}: {acc:.2f} +- {se:.2f}")


def plot_selected_window_sizes(run_results, vlines=[]):
    chosen_window_sizes = [[] for _ in run_results[0]["exp_results"]]
    for run_result_ in run_results:
        run_result = run_result_["exp_results"]
        for i in range(len(run_result)):
            chosen_window_sizes[i].append(run_result[i]["best_ws"])
    chosen_window_sizes = [np.mean(item) for item in chosen_window_sizes]

    min_y = min(chosen_window_sizes)
    max_y = max(chosen_window_sizes)

    plt.style.use(["science", "ieee"])
    fig, ax = plt.subplots(layout="tight", figsize=(3, 2), dpi=600)
    ax.vlines(vlines, min_y, max_y, linestyles="dashed", colors=["slategrey"])
    ax.plot(list(range(len(chosen_window_sizes))), chosen_window_sizes, color="green")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Window Size")
    sns.despine()
    fig.savefig(f"./fig_files/awa2_window_size_per_timestep.png")
    fig.savefig(f"./fig_files/awa2_window_size_per_timestep.pdf")


def plot_overall_acc(run_results, plot_mv=True):
    window_sizes = run_results[0]["window_sizes"]
    acc_list = {ws: [] for ws in window_sizes + ["dynamic", "majority_vote", "max"]}
    se_list = {ws: [] for ws in window_sizes + ["dynamic", "majority_vote", "max"]}

    best_window_sizes = list()

    for run_result_ in run_results:
        run_result = run_result_["exp_results"]
        ws_res = {ws: [] for ws in window_sizes + ["dynamic", "majority_vote", "max"]}
        for res in run_result:
            for ws, is_correct in res.items():
                if ws in ws_res.keys():
                    ws_res[ws].append(is_correct)
            best_window_sizes.append(res["best_ws"])
        for ws, correct_list in ws_res.items():
            if len(correct_list) != 0:
                acc_list[ws].append(np.mean(correct_list))
                se_list[ws].append(scipy.stats.sem(correct_list))
    valid_window_sizes = list(
        sorted(
            [
                ws
                for ws, val in acc_list.items()
                if len(val) != 0 and ws not in ["dynamic", "majority_vote", "max"]
            ]
        )
    )

    if len(run_results) == 1:
        dynamic_acc = [acc_list["dynamic"][0]] * len(valid_window_sizes)
        dynamic_se = [se_list["dynamic"][0]] * len(valid_window_sizes)
        max_acc = [acc_list["max"][0]] * len(valid_window_sizes)
        max_se = [se_list["max"][0]] * len(valid_window_sizes)
        mv_acc = [acc_list["majority_vote"][0]] * len(valid_window_sizes)
        mv_se = [se_list["majority_vote"][0]] * len(valid_window_sizes)
        ws_acc = [acc_list[ws][0] for ws in valid_window_sizes]
        ws_se = [se_list[ws][0] for ws in valid_window_sizes]
    else:
        dynamic_acc = [np.mean(acc_list["dynamic"])] * len(valid_window_sizes)
        dynamic_se = [scipy.stats.sem(acc_list["dynamic"])] * len(valid_window_sizes)
        max_acc = [np.mean(acc_list["max"])] * len(valid_window_sizes)
        max_se = [scipy.stats.sem(acc_list["max"])] * len(valid_window_sizes)
        mv_acc = [np.mean(acc_list["majority_vote"])] * len(valid_window_sizes)
        mv_se = [scipy.stats.sem(acc_list["majority_vote"])] * len(valid_window_sizes)
        ws_acc = [np.mean(acc_list[ws]) for ws in valid_window_sizes]
        ws_se = [scipy.stats.sem(acc_list[ws]) for ws in valid_window_sizes]

    dynamic_acc = [a * 100 for a in dynamic_acc]
    dynamic_se = [a * 100 for a in dynamic_se]
    max_acc = [a * 100 for a in max_acc]
    max_se = [a * 100 for a in max_se]
    mv_acc = [a * 100 for a in mv_acc]
    mv_se = [a * 100 for a in mv_se]
    ws_acc = [a * 100 for a in ws_acc]
    ws_se = [a * 100 for a in ws_se]

    print_accuracies(
        valid_window_sizes,
        dynamic_acc[0],
        dynamic_se[0],
        ws_acc,
        ws_se,
        max_acc[0],
        max_se[0],
    )

    best_ws_counter = dict(collections.Counter(best_window_sizes))
    counted_ws = list()
    ws_counts = list()
    for ak in sorted(list(best_ws_counter.keys())):
        counted_ws.append(str(ak))
        ws_counts.append(best_ws_counter[ak] / len(run_results))

    plt.style.use(["science", "ieee"])
    fig1, ax1 = plt.subplots(layout="tight", figsize=(3, 2), dpi=600)
    ax1.errorbar(x=valid_window_sizes, y=dynamic_acc, yerr=dynamic_se, label="Adaptive", linestyle='-', color='green')
    if plot_mv:
        ax1.errorbar(x=valid_window_sizes, y=mv_acc, yerr=mv_se, label="Majority Vote", linestyle='-')
    ax1.errorbar(x=valid_window_sizes, y=ws_acc, yerr=ws_se, label="Fixed", linestyle='-', color='blue')
    ax1.legend(loc="best")
    ax1.set_xlabel("Window Size")
    ax1.set_ylabel("Accuracy")
    # ax1.set_title("Accuracy vs. Window Size")
    sns.despine()


    Path("./fig_files").mkdir(exist_ok=True, parents=True)

    fig1.savefig("./fig_files/accuracy_vs_window_size.png")
    fig1.savefig("./fig_files/accuracy_vs_window_size.pdf")

    fig2, ax2 = plt.subplots(layout="tight", figsize=(3, 2), dpi=600)
    ax2.bar(counted_ws, ws_counts)
    ax2.set_xlabel("Window Size")
    ax2.set_ylabel("Count")
    # ax2.set_title("Dynamically Selected Window Sizes")
    sns.despine()

    fig2.savefig("./fig_files/selected_window_sizes.png")
    fig2.savefig("./fig_files/selected_window_sizes.pdf")


def avg_with_exclude(arr, avg_len):
    arr = np.asarray(arr)
    if np.sum(arr != -1) != avg_len:
        return None
    return np.mean(arr[arr != -1]) * 100


def plot_subsection(
    exp_results,
    run_idx=0,
    start=5000,
    end=None,
    plot_ws=["dynamic"],
    avg_len=200,
    jump_len=1,
    vlines=[]
):
    run_result = exp_results[run_idx]["exp_results"]
    all_ws = exp_results[run_idx]["window_sizes"]

    if end is None:
        end = len(run_result)

    pred_lists = {ws: [] for ws in all_ws + ["dynamic", "majority_vote", "max"]}

    for res in run_result:
        for ws in pred_lists.keys():
            pred_lists[ws].append(res.get(ws, -1))

    acc_list = {ws: [] for ws in plot_ws}
    acc_list_timesteps = {ws: [] for ws in acc_list.keys()}
    for idx in range(start, end - avg_len, jump_len):
        for ws in acc_list.keys():
            if isinstance(ws, int) and idx + 1 < ws:
                continue
            ts_acc = avg_with_exclude(
                pred_lists[ws][idx : idx + avg_len], avg_len=avg_len
            )
            if ts_acc is not None:
                acc_list[ws].append(ts_acc)
                acc_list_timesteps[ws].append(idx)

    name_mapping = {
        "majority_vote": "Majority Vote",
        "dynamic": "Adaptive",
        "max": "Max",
    }
    acc_keys = sorted([i for i in plot_ws if not isinstance(i, str)])
    if "majority_vote" in plot_ws:
        acc_keys = acc_keys + ["majority_vote"]
    if "dynamic" in plot_ws:
        acc_keys = acc_keys + ["dynamic"]

    plotting_values = list()
    for k in acc_keys:
        plotting_values.append(
            dict(
                x=acc_list_timesteps[k],
                y=acc_list[k],
                label=name_mapping.get(k, str(k)),
            )
        )

    Path("./fig_files").mkdir(exist_ok=True, parents=True)
    draw_plot(
        plotting_values, path=f"./fig_files/accuracy_over_time_{run_idx}", vlines=vlines
    )


def draw_plot(values, path, vlines):
    min_y = min([min(v["y"]) for v in values])
    max_y = max([max(v["y"]) for v in values])

    plt.style.use(["science", "ieee"])
    fig, ax = plt.subplots(layout="tight", figsize=(6, 2.5), dpi=600)
    ax.set_prop_cycle(color=["r", "b", "g", "k"], ls=["--", ":", "-.", "-"])
    for data in values:
        ax.plot(data["x"], data["y"], label=data["label"])
    ax.vlines(vlines, min_y, max_y, linestyles="dashed", colors=["slategrey"])
    plt.legend(loc="best")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Accuracy")
    sns.despine()
    fig.savefig(path + ".png")
    fig.savefig(path + ".pdf")
