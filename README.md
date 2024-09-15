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