from parse_raw_tsv import parse_from_tsv

def generate_vbs_resdict(all_result_dict):
    """
    generate virtual best solver result dictionary from the all-result dictionary
    vbs result dictionary: benchmark -> (is_solved, is_correct, status_str, cputime)
    """
    vbs_result_dict = {}
    for bench_path, config_results in all_result_dict.items():
        best_config = -1
        best_time = float('inf')
        for i, config_result in enumerate(config_results):
            is_solved, is_correct, status_str, cputime = config_result
            if is_solved and is_correct and cputime < best_time:
                best_config = i
                best_time = cputime
        # default output if no config can solve: (False, False, 'UNKNOWN', 0); other than is_solved, other entries have no meaning
        bench_vbs_result = config_results[best_config] if best_config != -1 else (False, False, 'UNKNOWN', 0)
        vbs_result_dict[bench_path] = bench_vbs_result
    return vbs_result_dict

def generate_vbs_tooldicts(all_result_dict):
    vbs_tool2bench_dict = {}
    vbs_bench2tool_dict = {}
    for bench_path, config_results in all_result_dict.items():
        best_config = -1
        best_time = float('inf')
        for i, config_result in enumerate(config_results):
            is_solved, is_correct, status_str, cputime = config_result
            if is_solved and is_correct and cputime < best_time:
                best_config = i
                best_time = cputime
        if best_config != -1:
            if best_config not in vbs_tool2bench_dict:
                vbs_tool2bench_dict[best_config] = []
            vbs_tool2bench_dict[best_config].append(bench_path)
            vbs_bench2tool_dict[bench_path] = best_config
    return vbs_tool2bench_dict, vbs_bench2tool_dict

def generate_one_resdict(all_result_dict, config_idx, prefix=None):
    """
    generate one tool-config result dictionary from the all-result dictionary
    one result dictionary: benchmark -> (is_solved, is_correct, status_str, cputime)
    """
    one_result_dict = {}
    for bench_path, config_results in all_result_dict.items():
        if prefix is not None and not bench_path.startswith(prefix):
            continue
        one_result_dict[bench_path] = config_results[config_idx]
    return one_result_dict

def is_dominating(all_res_dict, config_idx1, config_idx2, eval_func):
    """
    check whether config_idx1 is dominating config_idx2
    return 1 if config_idx1 is dominating config_idx2
    return -1 if config_idx2 is dominating config_idx1
    return 0 if neither is dominating
    """
    dom_1_2 = True
    dom_2_1 = True
    for bench_path, config_results in all_res_dict.items():
        res1 = config_results[config_idx1]
        res2 = config_results[config_idx2]
        eval1 = eval_func(res1)
        eval2 = eval_func(res2)
        if eval1 > eval2:
            dom_1_2 = False
        if eval1 < eval2:
            dom_2_1 = False
    if dom_1_2: # in cases of equality, we consider config_idx1 is dominating config_idx2
        return 1
    if dom_2_1:
        return -1
    return 0

def is_contribute_to_vbs(all_res_dict, config_idx, eval_func): 
    """
    check whether config_idx contributes to the virtual best solver
    here assume that eval score the lower the better
    """
    for bench_path, config_results in all_res_dict.items():
        # print(f"bench_path: {bench_path}")
        config_idx_res = config_results[config_idx]
        config_idx_score = eval_func(config_idx_res)
        is_contribute_bench = True
        for i, res in enumerate(config_results):
            res_score = eval_func(res)
            if config_idx_score >= res_score and i != config_idx:
                # print(f"config_idx_score: {config_idx_score}, res_score: {res_score}")
                is_contribute_bench = False
        # print("\n")
        if is_contribute_bench: return True
    return False

def is_correctly_solved(result_tuple: tuple[bool, bool, str, float]) -> bool:
    """
    check whether a tool-config result tuple is correctly solved
    """
    is_solved, is_correct, status_str, cputime = result_tuple
    return is_solved and is_correct

def is_incorrectly_solved(result_tuple: tuple[bool, bool, str, float]) -> bool:
    """
    check whether a tool-config result tuple is incorrectly solved
    """
    is_solved, is_correct, status_str, cputime = result_tuple
    return is_solved and not is_correct

def get_solved_num_for_dict(result_dict):
    """
    get the number of correctly solved benchmarks from a tool-config result dictionary
    """
    return sum([is_correctly_solved(result_tuple) for result_tuple in result_dict.values()])

def get_incorrect_num_for_dict(result_dict):
    """
    get the number of incorrectly solved benchmarks from a tool-config result dictionary
    """
    # for bench_path, result_tuple in result_dict.items():
    #     if is_incorrectly_solved(result_tuple):
    #         print(f"{bench_path}: {result_tuple}")
    return sum([is_incorrectly_solved(result_tuple) for result_tuple in result_dict.values()])

def get_par_N(result_tuple: tuple[bool, bool, str, float], N: int, timeout: float) -> float:
    """
    get the PAR-N score from a tool-config result tuple
    PAR-N: runtime if solved else N x timeout
    """
    is_solved, is_correct, status_str, cputime = result_tuple
    return cputime if (is_solved and is_correct) else N * timeout

def get_par2_err(result_tuple: tuple[bool, bool, str, float], timeout: float) -> float:
    """
    get the PAR-2 score with a 10x timeout penalty for errorous results
    """
    is_solved, is_correct, status_str, cputime = result_tuple
    if is_solved:
        if is_correct:
            return cputime
        else:
            return 10 * timeout
    return 2 * timeout

def get_par_N_for_dict(result_dict, N, timeout):
    """
    get the PAR-N score from a tool-config result dictionary
    PAR-N: runtime if solved else N x timeout
    """
    return sum([get_par_N(result_tuple, N, timeout) for bench_path, result_tuple in result_dict.items()])

def get_sbs(result_dict, dict_eval_func):
    """
    get the single best tool-config from a tool-config result dictionary
    Assume the lower the eval_func score, the better
    """
    assert len(result_dict) > 0
    first_res = list(result_dict.values())[0]
    tool_config_size = len(first_res)
    best_config = -1
    best_score = float('inf')
    for i in range(tool_config_size):
        one_score = dict_eval_func(generate_one_resdict(result_dict, i))
        if one_score < best_score:
            best_config = i
            best_score = one_score
    assert best_config != -1
    return best_config

def main():
    TIMEOUT = 900
    file_path = '/Users/zhengyanglu/Desktop/btor2select/bv_data/preprocessed_bv_results_0623.table.csv'
    result_dict, tool_config_dict = parse_from_tsv(file_path)
    tool_config_num = len(tool_config_dict)
    benchmark_size = len(result_dict)
    single_best_config = -1
    single_best_par2 = float('inf')
    for i in range(tool_config_num):
        one_resdict = generate_one_resdict(result_dict, i)
        one_solved = get_solved_num_for_dict(one_resdict)
        one_par2 = get_par_N_for_dict(one_resdict, 2, TIMEOUT)
        print(f"tool-config {i}: {one_solved} solved, PAR-2: {one_par2:.1f} ({tool_config_dict[i][1]})")
        if one_par2 < single_best_par2:
            single_best_config = i
            single_best_par2 = one_par2
    print()
    assert single_best_config != -1
    single_best_solved = get_solved_num_for_dict(generate_one_resdict(result_dict, single_best_config))
    print(f"Single Best: {tool_config_dict[single_best_config][1]}/{benchmark_size}, {single_best_solved} solved, PAR-2: {single_best_par2:.1f}")
    vbs_resdict = generate_vbs_resdict(result_dict)
    vbs_solved = get_solved_num_for_dict(vbs_resdict)
    vbs_par2 = get_par_N_for_dict(vbs_resdict, 2, TIMEOUT)
    print(f"Virtual Best: {vbs_solved}/{benchmark_size} solved, PAR-2: {vbs_par2:.1f}")

if __name__ == '__main__':
    main()
