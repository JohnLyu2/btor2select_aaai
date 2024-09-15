This repo supports our AAAI'25 Student Program Submission "Algorithm Selection for Word-Level Hardware Model Checking".

## Requirements and Installation

### Python
We tested the program with Python 3.12.4. You can install the required Python dependencies using `requirement.txt`:
```bash
    pip install -r requirements.txt
```

### Compile the `counts` binary (counting Btor2 features)
```bash
cd btor2feature
./configure.sh
cd build
make
```

## Reproducibility
We provide a Jupyter Notebook `reproduce.ipynb` to interactively reproduce our results. 

## HWMCC'24 Submission
We submitted a sequential compositional verifier `Btor2-SelectMC` to [HWMCC'24](https://hwmcc.github.io/2024/) based on this work. Check our submission at this [Zenodo link](https://zenodo.org/records/13627812).