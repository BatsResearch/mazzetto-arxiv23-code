import pickle as pkl

import dynamic_algorithm
import labelmodel
import numpy as np
from tqdm import tqdm

import awa2_dataset
import plot_utils


def run_exp(streaming_args, starting_index=5000, seed=0):
    votes, labels = awa2_dataset.get_data(seed=seed)

    r_base = streaming_args.get("r_base", 2)
    window_sizes = [
        int(r_base**i) for i in range(100) if int(r_base**i) <= votes.shape[0]
    ]
    exp_results = []

    mv_weights = np.ones([votes.shape[1]])

    for i in tqdm(range(starting_index, len(labels))):
        exp_results.append({})
        train_votes = votes[: i + 1]
        test_votes = votes[i]
        test_label = labels[i]

        best_window_size = dynamic_algorithm.compute_best_history_range(
            train_votes,
            delta=streaming_args["delta"],
            beta=streaming_args["beta"],
            r_base=streaming_args["r_base"],
            max_ws=train_votes.shape[0],
        )["optimal_window_size"]
        exp_results[-1]["best_ws"] = best_window_size

        majority_pred = labelmodel.predict(test_votes[None, :], mv_weights)
        exp_results[-1]["majority_vote"] = majority_pred[0] == test_label

        for ws in window_sizes:
            weights = labelmodel.estimate_accuracies(
                train_votes[max(0, train_votes.shape[0] - ws) :]
            )
            pred = labelmodel.predict(test_votes[None, :], weights)
            is_correct = pred[0] == test_label
            exp_results[-1][ws] = is_correct

            if ws == best_window_size:
                exp_results[-1]["dynamic"] = is_correct

        max_weights = labelmodel.estimate_accuracies(train_votes)
        max_pred = labelmodel.predict(test_votes[None, :], max_weights)
        exp_results[-1]["max"] = max_pred[0] == test_label

    return {"exp_results": exp_results, "window_sizes": window_sizes}


def main(streaming_args, num_runs=1, starting_index=5000):
    run_results = list()

    for run_idx in range(num_runs):
        print(f"Run {run_idx}")
        res = run_exp(streaming_args, starting_index=starting_index, seed=run_idx)
        run_results.append(res)

    vlines = [4000, 8000, 12000, 16000]
    plot_utils.plot_overall_acc(run_results, plot_mv=False)
    for i in range(num_runs):
        plot_utils.plot_subsection(
            run_results,
            run_idx=i,
            start=0,
            end=None,
            plot_ws=[16, 256, 1024, "dynamic"],
            avg_len=256,
            jump_len=100,
            vlines=vlines,
        )

    plot_utils.plot_selected_window_sizes(run_results, vlines=vlines)

    return run_results


if __name__ == "__main__":
    streaming_args = {"delta": 0.1, "beta": 0.1, "r_base": 2}
    num_runs = 3
    starting_index = 0
    exp_results = main(streaming_args, num_runs=num_runs, starting_index=starting_index)

    summary = {
        "args": {
            "streaming_args": streaming_args,
            "num_runs": num_runs,
            "starting_index": starting_index,
        },
        "results": exp_results,
    }

    with open("exp_results.pkl", "wb") as f:
        pkl.dump(summary, f)
