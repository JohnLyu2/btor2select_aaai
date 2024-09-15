import subprocess
import sys
from pathlib import Path
import csv
import time
# from sklearn.preprocessing import MinMaxScaler

from parse_raw_tsv import get_btor2path_from_yml

COUNTS_BINARY = "btor2feature/build/bin/counts"

KEYWORDS = [
    "add", "and", "bad", "constraint", "concat", "const",
    "constd", "consth", "dec", "eq", "fair", "iff",
    "implies", "iff", "inc", "init", "input", "ite",
    "justice", "mul", "nand", "neq", "neg",
    "next", "nor", "not", "one", "ones", "or",
    "output", "read", "redand", "redor", "redxor", "rol",
    "ror", "saddo", "sext", "sgt", "sgte", "sdiv",
    "sdivo", "slice", "sll", "slt", "slte",
    "sort", "smod", "smulo", "ssubo", "sra", "srl",
    "srem", "state", "sub", "uaddo", "udiv", "uext",
    "ugt", "ugte", "ult", "ulte", "umulo", "urem",
    "usubo", "write", "xnor", "xor", "zero"
]

def parse_counts(benchmark_path: str) -> list[int]:
    """
    Parse results from counts_binary for a btor2 benchmark (in btor2)
    """
    benchmark_path = Path(benchmark_path)
    if not benchmark_path.is_file():
        raise ValueError(f"The provided benchmark does not exist: {benchmark_path}")
    # check whether counts_binary exists
    if not Path(COUNTS_BINARY).is_file():
        raise ValueError("The btor2kwcount binary does not exist. Please build it first.")
    btor2counts_output = subprocess.run([COUNTS_BINARY, benchmark_path], capture_output=True, text=True)
    if btor2counts_output.returncode != 0:
        raise ValueError(f"Error processing {benchmark_path}: {btor2counts_output.stderr}")
    outputlines = btor2counts_output.stdout.splitlines()
    assert len(outputlines) == 2, f"Unexpected output from {benchmark_path}: {outputlines}"
    # kwcount output is in the format: kw0count kw1count ... kwNcount
    kw_counts = outputlines[0].split()
    kw_counts = [int(count) for count in kw_counts]
    # make sure the number of keywords is correct
    if len(kw_counts) != len(KEYWORDS):
        raise ValueError(f"Unexpected number {len(kw_counts)} of keywords in {benchmark_path} embedding")
    bit_counts = outputlines[1].split()
    bit_counts = [int(count) for count in bit_counts]
    return kw_counts, bit_counts

def get_kwcounts(benchmark_path: str) -> list[int]:
    """
    Get the keyword counts for a btor2 benchmark (in btor2)
    """
    kw_counts, _ = parse_counts(benchmark_path)
    return kw_counts

def get_bitcounts(benchmark_path: str) -> list[int]:
    """
    Get the bit counts for a btor2 benchmark (in btor2)
    """
    _, bit_counts = parse_counts(benchmark_path)
    return bit_counts

def get_feature_v1_kwcount(benchmark_path: str) -> list[int]:
    """
    Create a V1 feature based on keyword counts for a btor2 benchmark (in btor2)
    V1 feature: keywords + uninit_states
    """
    # init is index 15, input is index 16, sort is index 46, state is index 53
    kwcounts, _ = parse_counts(benchmark_path)
    uninit_states = kwcounts[53] - kwcounts[15]
    return kwcounts + [uninit_states]

def get_feature_v1_bits(benchmark_path: str) -> list[int]:
    """
    Create a V1 feature based on bit sums for a btor2 benchmark (in btor2)
    V1 feature: keywords + uninit_states
    """
    # init is index 15, input is index 16, sort is index 46, state is index 53
    _, bitcounts = parse_counts(benchmark_path)
    uninit_state_bits = bitcounts[53] - bitcounts[15]
    return bitcounts + [uninit_state_bits]


def create_btor2kw_dict(benchmark_dir: str) -> dict[str, tuple[list[int], float]]:
    """
    Create a dictionary for the btor2kw embeddings and the processing time of each btor2 benchmark (yml) in the benchmark directory 
    """
    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.is_dir():
        raise ValueError(f"The provided benchmark directory does not exist: {benchmark_dir}")
    # check whether btor2kwcount/build/bin/kwcount binary exists
    if not Path(COUNTS_BINARY).is_file():
        raise ValueError("The btor2 counts binary does not exist. Please build it first.")
    output_dict = {}
    for btor2_yml in benchmark_dir.rglob("*.yml"):
        btor2_file = get_btor2path_from_yml(btor2_yml)
        btor2_file = str(btor2_file)
        start_time = time.time()
        kw_counts = get_kwcounts(btor2_file)
        end_time = time.time()
        output_dict[btor2_yml] = (end_time - start_time, kw_counts)
    return output_dict

def create_btor2kw_csv(benchmark_dir: str, output_path: str):
    """
    Create a CSV file for the btor2kw embeddings of each btor2 benchmark in the benchmark directory
    """
    output_file = Path(output_path)
    # Create the output directory if it does not exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    btor2_dict = create_btor2kw_dict(benchmark_dir)
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["benchmark"]+ ["time"] + KEYWORDS)
        for btor2_yml, (time, kw_counts) in btor2_dict.items():
            csv_writer.writerow([btor2_yml]+ [time] + kw_counts)

def read_btor2kw_csv(csv_path: str) -> dict[str, tuple[float, list[int]]]:
    """
    Read the btor2kw embeddings into a dictionary from a CSV file
    """
    btor2kw_dict = {}
    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        if header[1:] != ["time"] + KEYWORDS:
            raise ValueError(f"Unexpected header in {csv_path}")
        for row in csv_reader:
            btor2kw_dict[row[0]] = (float(row[1]), [int(count) for count in row[2:]])
    return btor2kw_dict

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python create_btor2kw.py <benchmark_dir> <output_csv>")
    #     sys.exit(1)
    # create_btor2kw_csv(sys.argv[1], sys.argv[2])
    print(get_feature_v1_kwcount("/Users/zhengyanglu/Desktop/btor2select_material/word-level-hwmc-benchmarks/bv/btor2/btor2tools-examples/factorial4even.btor2"))
    print(get_feature_v1_bits("/Users/zhengyanglu/Desktop/btor2select_material/word-level-hwmc-benchmarks/bv/btor2/btor2tools-examples/factorial4even.btor2"))
    print()
    print(get_feature_v1_kwcount("/Users/zhengyanglu/Desktop/btor2select_material/word-level-hwmc-benchmarks/bv/btor2/beem/adding.2.prop1-back-serstep.btor2"))
    print(get_feature_v1_bits("/Users/zhengyanglu/Desktop/btor2select_material/word-level-hwmc-benchmarks/bv/btor2/beem/adding.2.prop1-back-serstep.btor2"))
    # print(kw_embed)