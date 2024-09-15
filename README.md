This repo supports our AAAI'25 Student Program Submission "Algorithm Selection for Word-Level Hardware Model Checking".

## Requirements and Installation

#### Python
We tested the program with Python 3.12.4. You can install the required Python dependencies using `requirement.txt`:
```bash
pip install -r requirements.txt
```

#### Compile the `counts` binary (counting Btor2 features)
```bash
cd btor2feature
./configure.sh
cd build
make
```

## Reproducibility
We provide a Jupyter Notebook `reproduce.ipynb` to interactively reproduce our results. The performance data are stored in `performance_data/performance.table.csv`. They were collected from verifier-instance executions on Ubuntu 22.04 machines, each with a 3.4 GHz CPU (Intel Xeon E3-1230 v5) with 8 processing units and 33 GB of RAM. Each task was assigned 2 CPU cores, 15 GB RAM, and 15 min of CPU time limit. We used [BenchExec](https://github.com/sosy-lab/benchexec) to ensure reliable resource measurement and reproducible results.  

## HWMCC'24 Submission
We submitted a sequential compositional verifier `Btor2-SelectMC` to [HWMCC'24](https://hwmcc.github.io/2024/) based on this work. Check our submission at this [Zenodo link](https://zenodo.org/records/13627812).

## License
Btor2-Select is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). The submodule `counts` is largely based on codes from [Btor2Tools](https://github.com/Boolector/btor2tools), which is licensed under the [MIT License](btor2kwcount/LICENSE.txt).