import itertools
import json

import numpy as np
import pandas as pd


def get_attr_value_vote_mapping(class_attr_matrix, lf_names, classes_0, classes_1):
    classes_flat_0 = classes_0[0] + classes_0[1]
    classes_flat_1 = classes_1[0] + classes_1[1]
    attr_value_votes = {}
    for attr in lf_names:
        values_0 = [class_attr_matrix[c][attr] for c in classes_flat_0]
        values_1 = [class_attr_matrix[c][attr] for c in classes_flat_1]
        assert len(set(values_0)) == 1 and len(set(values_1)) == 1
        assert values_0[0] != values_1[0]
        attr_value_votes[attr] = {values_0[0]: 0, values_1[0]: 1}
    return attr_value_votes


def get_super_label(animal, classes_0, classes_1):
    classes_0_flat = list(itertools.chain(*classes_0))
    classes_1_flat = list(itertools.chain(*classes_1))
    if animal in classes_0_flat:
        return 0
    if animal in classes_1_flat:
        return 1
    raise RuntimeError


def load_data(classes_0, classes_1, lf_names):
    all_classes = list(itertools.chain(*classes_0, *classes_1))
    with open("./class_attr_matrix.json", "r") as f:
        class_attr_mapping = json.load(f)
    attr_value_vote_mapping = get_attr_value_vote_mapping(
        class_attr_matrix=class_attr_mapping,
        lf_names=lf_names,
        classes_0=classes_0,
        classes_1=classes_1,
    )

    attr_detectors = pd.read_csv("./attr_detectors.csv", sep=",").to_dict("records")

    data_list = list()
    for example_attrs in attr_detectors:
        animal = example_attrs["animal"]
        if animal not in all_classes:
            continue
        animal_super_label = get_super_label(
            animal=animal, classes_0=classes_0, classes_1=classes_1
        )
        rec_ = {
            attr: attr_value_vote_mapping[attr][example_attrs[attr]]
            for attr in lf_names
        }
        rec_["animal"] = animal
        rec_["super_label"] = animal_super_label
        data_list.append(rec_)
    df = pd.DataFrame(data_list)

    votes_arr = df[lf_names].to_numpy(copy=True)
    labels_arr = df["super_label"].to_numpy(copy=True)
    animal_list = df["animal"].tolist()
    per_animal_indices = {a: list() for a in list(set(animal_list))}
    for a_idx in range(len(animal_list)):
        per_animal_indices[animal_list[a_idx]].append(a_idx)
    return votes_arr, labels_arr, per_animal_indices


def get_data(seed=0):
    np.random.seed(seed)

    classes_0 = [["horse"], ["sheep"]]
    classes_1 = [["giraffe"], ["bobcat"]]
    lf_names = ["black", "white", "orange", "yellow", "spots", "domestic"]

    votes_arr, labels_arr, per_animal_indices = load_data(
        classes_0=classes_0, classes_1=classes_1, lf_names=lf_names
    )

    class_00_indices = list(
        itertools.chain(*[per_animal_indices[k] for k in classes_0[0]])
    )
    class_01_indices = list(
        itertools.chain(*[per_animal_indices[k] for k in classes_0[1]])
    )
    class_10_indices = list(
        itertools.chain(*[per_animal_indices[k] for k in classes_1[0]])
    )
    class_11_indices = list(
        itertools.chain(*[per_animal_indices[k] for k in classes_1[1]])
    )

    period_size_and_ratio = [
        (4000, 0.1),
        (4000, 0.9),
        (4000, 0.1),
        (4000, 0.9),
        (4000, 0.1),
    ]

    selected_indices = list()

    for period_size, ratio in period_size_and_ratio:
        for _ in range(period_size):
            super_class = np.random.binomial(1, 0.5)
            subclass = np.random.binomial(1, ratio)

            if super_class == 0 and subclass == 0:
                c_idx = np.random.choice(class_00_indices)
            elif super_class == 0 and subclass == 1:
                c_idx = np.random.choice(class_01_indices)
            elif super_class == 1 and subclass == 0:
                c_idx = np.random.choice(class_10_indices)
            elif super_class == 1 and subclass == 1:
                c_idx = np.random.choice(class_11_indices)
            else:
                raise RuntimeError
            selected_indices.append(c_idx)

    votes_arr = votes_arr[selected_indices]
    labels_arr = labels_arr[selected_indices]

    votes_arr[votes_arr == 0] = -1
    labels_arr[labels_arr == 0] = -1

    assert set(np.unique(votes_arr).tolist()) == set([-1, 1])
    assert set(np.unique(labels_arr).tolist()) == set([-1, 1])

    return votes_arr, labels_arr


if __name__ == "__main__":
    output = get_data()
    votes = output["votes"]
    labels = output["labels"]
