import numpy as np
from pathlib import Path
from functools import partial
import joblib
import argparse

from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.utils.validation import check_X_y
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from parse_raw_tsv import parse_from_tsv, parse_from_tsv_lst, get_btor2path_from_yml
from create_feature import get_feature_v1_kwcount, get_feature_v1_bits, get_kwcounts, get_bitcounts
from analyze import get_par2_err, get_par_N

def generate_training_samples_for_config_pair(training_res_dict, tool_config_pair, res2target_func, feature_func):
    tool_config0, tool_config1 = tool_config_pair
    inputs = []
    labels = []
    costs = []
    for yml_path, result_lst in training_res_dict.items():
        btor2_path = get_btor2path_from_yml(yml_path)
        btor2kw = feature_func(btor2_path)
        result_tuple0 = result_lst[tool_config0]
        result_tuple1 = result_lst[tool_config1]
        target0 = res2target_func(result_tuple0)
        target1 = res2target_func(result_tuple1)
        # here assume always the less target the better
        label = 1 if target0 < target1 else 0 # label 1 represents tool_config0 is better
        cost = abs(target0 - target1)
        if cost > 1e-10: # if the performance difference is 0, ignore
            inputs.append(btor2kw)
            labels.append(label)
            costs.append(cost)
    inputs_array = np.array(inputs)
    labels_array = np.array(labels)
    costs_array = np.array(costs)
    return inputs_array, labels_array, costs_array, None, None

def generate_training_samples_for_config_pair_with_pca(training_res_dict, tool_config_pair, res2target_func, feature_func):
    inputs_array, labels_array, costs_array, _, _ = generate_training_samples_for_config_pair(training_res_dict, tool_config_pair, res2target_func, feature_func)
    scaler = StandardScaler()
    inputs_array = scaler.fit_transform(inputs_array)
    pca = PCA(n_components=0.95)
    inputs_array = pca.fit_transform(inputs_array)
    return inputs_array, labels_array, costs_array, scaler, pca

class PairwiseDecisionTree(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, weights):
        X, y = check_X_y(X, y)
        super().fit(X, y, sample_weight=weights)
        return self

class PairwiseXGBoost(xgb.XGBClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, weights):
        X, y = check_X_y(X, y)
        super().fit(X, y, sample_weight=weights)
        return self

def get_pw_algorithm_selection_lst(test_benchmark, model_matrix, feature_func, random_seed = 0, scaler_matrix = None, pca_matrix = None, exclude_lst = None):
    btor2_path = get_btor2path_from_yml(test_benchmark)
    btor2kw = feature_func(btor2_path)
    btor2kw_array = np.array(btor2kw).reshape(1, -1)
    # if scaler is not None and pca is not None:
    #     btor2kw_array = scaler.transform(btor2kw_array)
    #     btor2kw_array = pca.transform(btor2kw_array)
    config_size = model_matrix.shape[0]
    votes = np.zeros(config_size, dtype=int)
    for i in range(config_size):
        if exclude_lst is not None and i in exclude_lst:
            continue
        for j in range(i+1, config_size):
            if exclude_lst is not None and j in exclude_lst:
                continue
            if scaler_matrix is not None and pca_matrix is not None:
                btor2kw_i_j = scaler_matrix[i, j].transform(btor2kw_array)
                btor2kw_i_j = pca_matrix[i, j].transform(btor2kw_i_j)
            else:
                btor2kw_i_j = btor2kw_array
            prediction = model_matrix[i, j].predict(btor2kw_i_j)
            if prediction[0]: # i is better
                votes[i] += 1
            else:
                votes[j] += 1
    np.random.seed(random_seed)
    random_tiebreaker = np.random.random(config_size)
    structured_votes = np.rec.fromarrays([votes, random_tiebreaker], names='votes, random_tiebreaker')
    sorted_indices = np.argsort(structured_votes, order=('votes', 'random_tiebreaker'))[::-1]
    return sorted_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bit', action='store_true', help="Flag to use the BWA embedding")
    parser.add_argument('--pca', action='store_true', help="Flag to use PCA")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the trained models")
    parser.add_argument('input_files', nargs='+', help="List of input performance result files")
    args = parser.parse_args()

    bwa_flag = args.bit
    pca_flag = args.pca
    save_dir = args.save_dir
    input_files = args.input_files
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    feature_func = get_bitcounts if bwa_flag else get_kwcounts
    training_res_dict, tool_config_dict = parse_from_tsv_lst(input_files, num_tool_cols=2, timeout=900.0)
    tool_config_size = len(tool_config_dict)
    # par2_err_900 = partial(get_par2_err, timeout=900.0)
    par2_func = partial(get_par_N, N=2, timeout=900.0)
    generate_samples_func = generate_training_samples_for_config_pair_with_pca if pca_flag else generate_training_samples_for_config_pair

    for i in range(tool_config_size):
        for j in range(i+1, tool_config_size):
            inputs_array, labels_array, costs_array, scaler, pca = generate_samples_func(training_res_dict, (i, j), par2_func, feature_func)
            xm = PairwiseXGBoost()
            xm.fit(inputs_array, labels_array, costs_array)
            joblib.dump(xm, f"{save_dir}/xg_{i}_{j}.joblib")
            if pca_flag:
                joblib.dump(scaler, f"{save_dir}/scaler_{i}_{j}.joblib")
                joblib.dump(pca, f"{save_dir}/pca_{i}_{j}.joblib")
            print(f"Finished training model for ({i}, {j})")

if __name__ == "__main__":
    main()
