from parse_raw_tsv import parse_from_tsv_lst, parse_from_tsv, get_btor2path_from_yml
from create_feature import get_kwcounts, get_bitcounts
from analyze import is_correctly_solved, get_par_N
from functools import partial
import numpy as np
from pathlib import Path
import argparse
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

def generate_input_target_for_one_config(training_res_dict, tool_config_id, res2target_func, feature_func):
    """
    generate training samples for one tool-config from the training result dictionary
    """
    inputs = []
    targets = []
    for yml_path, result_lst in training_res_dict.items():
        btor2_path = get_btor2path_from_yml(yml_path)
        features = feature_func(btor2_path)
        result_tuple = result_lst[tool_config_id]
        target = res2target_func(result_tuple)
        inputs.append(features)
        targets.append(target)
    inputs_array = np.array(inputs)
    targets_array = np.array(targets)
    return inputs_array, targets_array, None, None

def generate_input_target_for_one_config_with_pca(training_res_dict, tool_config_id, res2target_func, feature_func):
    inputs_array, targets_array, _, _ = generate_input_target_for_one_config(training_res_dict, tool_config_id, res2target_func, feature_func)
    scaler = StandardScaler()
    inputs_array = scaler.fit_transform(inputs_array)
    pca = PCA(n_components=0.95)
    inputs_array = pca.fit_transform(inputs_array)
    return inputs_array, targets_array, scaler, pca

def get_ehm_algorithm_selection_lst(benchmark, models, feature_func):
    btor2_path = get_btor2path_from_yml(benchmark)
    features = feature_func(btor2_path)
    features_array = np.array(features).reshape(1, -1)
    predictions = []
    for i in range(len(models)):
        model = models[i]
        output = model.predict(features_array)
        predictions.append((i, output.item()))  # Store both the model ID and the prediction
    # Sort the predictions by the prediction values (second item in the tuple)
    sorted_predictions = sorted(predictions, key=lambda x: x[1])
    # Extract the sorted model IDs
    sorted_model_ids = [model_id for model_id, _ in sorted_predictions]
    return sorted_model_ids

def get_ehm_algorithm_selection_lst(benchmark, models, feature_func, scalers=None, pcas=None):
    btor2_path = get_btor2path_from_yml(benchmark)
    features = feature_func(btor2_path)
    features_array = np.array(features).reshape(1, -1)
    predictions = []
    for i in range(len(models)):
        model = models[i]
        if scalers is not None and pcas is not None:
            features_array_i = scalers[i].transform(features_array)
            features_array_i = pcas[i].transform(features_array_i)
        else:
            features_array_i = features_array
        output = model.predict(features_array_i)[0]
        predictions.append((i, output))  # Store both the model ID and the prediction
    # Sort the predictions by the prediction values (second item in the tuple)
    sorted_predictions = sorted(predictions, key=lambda x: x[1])
    # Extract the sorted model IDs
    sorted_model_ids = [model_id for model_id, _ in sorted_predictions]
    return sorted_model_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bit', action='store_true', help="Flag to use the BWA embedding")
    parser.add_argument('--pca', action='store_true', help="Flag to use PCA")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the trained models")
    parser.add_argument('input_files', nargs='+', help="List of input performance result files")
    args = parser.parse_args()

    bwa_flag = args.bit
    save_dir = args.save_dir
    input_files = args.input_files
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    feature_func = get_bitcounts if bwa_flag else get_kwcounts

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    training_res_dict, tool_config_dict = parse_from_tsv_lst(input_files, num_tool_cols=2, timeout=900.0)
    par2_func = partial(get_par_N, N=2, timeout=900.0)
    generate_samples_func = generate_input_target_for_one_config_with_pca if args.pca else generate_input_target_for_one_config
    for i in range(len(tool_config_dict)):
        inputs_array, targets_array, scaler, pca = generate_samples_func(training_res_dict, i, par2_func, feature_func)
        xgb_reg = xgb.XGBRegressor()
        xgb_reg.fit(inputs_array, targets_array)
        xgb_reg.save_model(f"{save_dir}/xg_{i}.json")
        if args.pca:
            joblib.dump(scaler, f"{save_dir}/scaler_{i}.joblib")
            joblib.dump(pca, f"{save_dir}/pca_{i}.joblib")
        print(f"Finished training model for {i}")

if __name__ == "__main__":
    main()
