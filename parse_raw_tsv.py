import csv
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def process_raw_result_tuple(raw_result_tuple: tuple[str, str], verify_labl: bool, bench_path: str) -> tuple[bool, bool, str, float]:
    """
    process the raw result tuple (status_str, cputime_str) to (is_solved, is_correct, status_str, cputime)
    bench_path: the benchmark path in raw tsv column 0; for error message
    assume the status_str and cputime_str are not empty; the empty cases are not handled here
    """
    status_str, cputime_str = raw_result_tuple
    assert status_str != '' or cputime_str != ''
    is_solved: bool = False
    is_correct: bool = True
    other_unsolved_labels = ['OUT OF MEMORY', 'EXCEPTION', 'done', 'SEGMENTATION FAULT', 'ABORTED']
    if status_str == 'true' or status_str == 'false' or status_str == 'false(unreach-call)':
        is_solved = True
        if status_str == 'true':
            if not verify_labl: is_correct = False # to-do: report incorrect results
        else:
            if verify_labl: is_correct = False
    # status_str starts with TIMEOUT or ERROR
    elif status_str.startswith('TIMEOUT') or status_str.startswith('ERROR') or status_str.startswith('KILLED') or status_str in other_unsolved_labels:
        pass
    else:
        raise ValueError(f"Unknown status value: \"{status_str}\" in {bench_path}")
    cputime: float = float(cputime_str)
    return is_solved, is_correct, status_str, cputime

def parse_from_tsv(file_path, num_tool_cols=2, timeout = 900.0, verdict = None):
    """
    parse from one tsv result file, outputs: bench_result_dict, tool_config_dict
    num_tool_cols: number of columns for each tool-config results; default is 2: status and cputime (s)
    assume the second column is verificaiton label
    only include results with "verdict" if not None
    """
    check_tsv_header(file_path)
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        header0: list[str] = next(reader)
        header1: list[str] = next(reader)
        header2: list[str] = next(reader)
        bench_top_dir: str = header2[0]
        num_tool_config: int = (len(header0) - 2)//num_tool_cols
        tool_config_dict: dict[int, tuple[str, str]] = {}
        # abc_pdr_status_col = -1
        for i in range(num_tool_config):
            tool_config_status_col: int = 2 + i*num_tool_cols
            assert header2[tool_config_status_col] == 'status'
            tool_name: str = header0[tool_config_status_col]
            config_name: str = header1[tool_config_status_col]
            if config_name == 'abc.pdr.bv64': abc_pdr_status_col = tool_config_status_col
            tool_config_dict[i] = (tool_name, config_name)
        assert abc_pdr_status_col != -1 # assume abc.pdr is always in the tool configs
        # print(f"Read {num_tool_config} tool-confg pairs")
        bench_result_dict: dict[str, list[tuple[bool, bool, str, float]]] = {}
        for row in reader:
            bench_subpath: str = row[0]
            bench_path: str = f"{bench_top_dir}{bench_subpath}"
            verify_label_str: str = row[1]
            if verify_label_str == 'true':
                verify_label: bool = True
            elif verify_label_str == 'false':
                verify_label: bool = False
            else:
                raise ValueError(f"Unknown verification label: \"{verify_label_str}\" in {bench_path}")
            if verdict is not None and verify_label != verdict: continue
            result_lst: list[tuple[bool, bool, str, float]] = []
            for i in range(num_tool_config):
                tool_status_col: int = 2 + i * num_tool_cols
                tool_status_str: str = row[tool_status_col]
                tool_cputime_str: str = row[tool_status_col + 1]
                raw_result_tuple: tuple[str, str] = (tool_status_str, tool_cputime_str)
                if raw_result_tuple == ('', ''):
                    # print(f"Empty result for {tool_config_dict[i][1]} in {bench_path}")
                    config_name = tool_config_dict[i][1]
                    if 'abc' in config_name:
                        if row[abc_pdr_status_col] == '': result_lst.append((False, True, 'UNKNOWN', 0.1)) # Btor2Aiger fails immediately in this case. So having CPU time close to 0s.
                        elif verify_label and ('false' in config_name): # the BMC configs were only executed on false tasks beacause they cannot find proofs
                            result_lst.append((False, True, 'TIMEOUT', timeout))
                        else:
                            raise ValueError(f"Unknown empty result for abc in {bench_path}")
                    else:
                        # the BMC configs were only executed on false tasks beacause they cannot find proofs
                        config_bmc_lst = ['avr.bmc-boolector', 'btormc.bmc'] # this list does not include configure names already having 'false' in them
                        if verify_label and ('false' in config_name or config_name in config_bmc_lst):
                            result_lst.append((False, True, 'TIMEOUT', timeout))
                        else:
                            # will there be any other cases that the result is missing?
                            # result_lst.append((False, True, 'UNKNOWN', 0.1)) 
                            raise ValueError(f"Unknown empty result for {tool_config_dict[i][1]} in {bench_path}")
                else:
                    processed_results = process_raw_result_tuple(raw_result_tuple, verify_label, bench_subpath)
                    result_lst.append(processed_results)
            bench_result_dict[bench_path] = result_lst
        return bench_result_dict, tool_config_dict

def parse_from_tsv_lst(file_lst, num_tool_cols=2, timeout = 900.0):
    """
    parse from a list of tsv result files and merge the data, outputs: bench_result_dict, tool_config_dict
    all tsv files shall have the same header (same tool-configs) as they are assumed to be partitioned from the same dataset
    num_tool_cols: number of columns for each tool-config results; default is 2: status and cputime (s)
    """
    assert len(file_lst) > 0
    all_bench_result_dict, tool_config_dict = parse_from_tsv(file_lst[0], num_tool_cols, timeout)
    for file_path in file_lst[1:]:
        bench_result_dict, _ = parse_from_tsv(file_path, num_tool_cols, timeout)
        for bench_path, result_lst in bench_result_dict.items():
            all_bench_result_dict[bench_path] = result_lst
    return all_bench_result_dict, tool_config_dict

def check_tsv_header(file_path: str):
    """
    check the header format of the tsv file
    """
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        header0: list[str] = next(reader)
        assert header0[0] == 'tool'
        assert header0[1] == ''
        header1: list[str] = next(reader)
        assert header1[0] == 'run set'
        assert header1[1] == ''
        header2: list[str] = next(reader)
        assert header2[1] == ''

def filter_by_prefix_and_rw_path(tsv_path: str, bench_prefix: str, output_path: str, local_parent_dir: str):
    """
    filter the tsv file by the bench prefix and rewrite the benchmark path
    this funciton is now customized for the provided tsv format
    it will insert "btor2/" after the bench_prefix; the bench_prefix is either "bv/" or 'array/' (or something else?)
    it also rewrites the first entry in the third row header to the local parent directory of the benchmark path
    save the filtered results to the output path
    """
    check_tsv_header(tsv_path)
    with open(tsv_path, 'r') as infile, open(output_path, 'w') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        for i, row in enumerate(reader):
            if i < 3:
                if i == 2: row[0] = local_parent_dir
                writer.writerow(row)
            else:
                bench_subpath = row[0]
                if bench_subpath.startswith(bench_prefix):
                    bench_subpath = bench_subpath.replace(bench_prefix, f"{bench_prefix}btor2/")
                    row[0] = bench_subpath
                    writer.writerow(row)

def filter_by_tool_configs(input_tsv_path, output_tsv_path, selected_tool_configs):
    """
    filter the tsv file by the selected tool configurations
    """
    check_tsv_header(input_tsv_path)
    with open(input_tsv_path, 'r') as infile, open(output_tsv_path, 'w') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        for row in reader:
            tool_config_res = row[2:]
            write_row = row[:2]
            for i in selected_tool_configs:
                write_row += tool_config_res[2*i:2*i+2]
            writer.writerow(write_row)
            
def tsv_N_fold_partition(tsv_path, N, output_dir, random_seed=33):
    """
    partition the all-tsv file into N parts, i.e., 0.csv, 1.csv, ..., N-1.csv
    The partitioned files are saved in the output directory
    """
    check_tsv_header(tsv_path)
    with open(tsv_path, 'r') as file:
        headers = [next(file) for _ in range(3)]
    # read the rest of the dataset into a pandas dataframe
    df = pd.read_csv(tsv_path, sep='\t', header=None, skiprows=3, dtype=str)
    # create the output directory if it does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # shuffle the dataframe randomly by the seed
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # partition the dataframe into N parts
    partitions = np.array_split(df, N)
    # Write each partition to a separate TSV file with headers
    for i, partition in enumerate(partitions):
        partition_path = output_dir / f"{i}.tsv"
        with open(partition_path, 'w', newline='') as csvfile:
            csvfile.writelines(headers)
        partition.to_csv(partition_path, sep='\t', header=False, index=False, mode='a')

def tsv_train_test_partition(tsv_path, train_to_test_ratio, output_dir, random_seed=33):
    """
    partition the all-tsv file into a training set and a test set with a train_to_test_ratio
    Now the ratio must be an integer
    The partitioned files are saved in the output directory
    """
    N = train_to_test_ratio + 1
    check_tsv_header(tsv_path)
    with open(tsv_path, 'r') as file:
        headers = [next(file) for _ in range(3)]
    # read the rest of the dataset into a pandas dataframe
    df = pd.read_csv(tsv_path, sep='\t', header=None, skiprows=3, dtype=str)
    # create the output directory if it does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # shuffle the dataframe randomly by the seed
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # partition the dataframe into N parts
    partitions = np.array_split(df, N)
    train_partitions = partitions[:-1]
    # concatenate the training partitions
    train_partition = pd.concat(train_partitions)
    test_partition = partitions[-1]
    # Write train and test to TSV file with headers
    train_path = output_dir / "train.tsv"
    with open(train_path, 'w', newline='') as csvfile:
        csvfile.writelines(headers)
    train_partition.to_csv(train_path, sep='\t', header=False, index=False, mode='a')
    test_path = output_dir / "test.tsv"
    with open(test_path, 'w', newline='') as csvfile:
        csvfile.writelines(headers)
    test_partition.to_csv(test_path, sep='\t', header=False, index=False, mode='a')

def get_btor2path_from_yml(yml_path: str) -> str:
    """
    get the btor2 path from the yml file
    """
    yml_path = Path(yml_path)
    if not yml_path.is_file():
        raise ValueError(f"The provided yml file does not exist: {yml_path}")
    with open(yml_path, 'r') as file:
        yml_dict = yaml.safe_load(file)
    btor2file = yml_dict['input_files']
    # get the btor2 file path: the parent directory of the yml file + the btor2 file name
    btor2path = yml_path.parent / btor2file
    return str(btor2path)

def main():
    # raw_tsv_path = '/Users/zhengyanglu/Desktop/btor2select/performance_data/performance-download_0623/no_arr_config_xml_download_0623.table.csv'
    # out_tsv_path = '/Users/zhengyanglu/Desktop/btor2select/bv_data/preprocessed_bv_results_0623.table.csv'
    # local_parent_dir = '/Users/zhengyanglu/Desktop/btor2select_material/word-level-hwmc-benchmarks/'
    # filter_by_prefix_and_rw_path(raw_tsv_path, 'bv/', out_tsv_path, local_parent_dir)
    # tsv_N_fold_partition('/Users/zhengyanglu/Desktop/btor2select/bv_data/0823/bv_0823.csv', 5, '/Users/zhengyanglu/Desktop/btor2select/bv_data/0823/5fold')
    # parse_from_tsv("/Users/zhengyanglu/Desktop/btor2select/bv_data/0823/bv_0823.csv")
    filter_by_tool_configs('bv_data/0823/bv_0823.csv', 'bv_data/0823/filter_bv_0823.csv', [0])

if __name__ == '__main__':
    main()