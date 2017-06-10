# pc_smac

SMAC for automatic algorithm selection and hyperparameter optimization of Scikit-learn machine learning pipelines. Inspired and based on [Auto-sklearn](https://github.com/automl/auto-sklearn). This version incorporates pipeline caching to speed up optimization.

## Intallation

Use python 3.6.

We recommend using an anaconda environment. Some commands on this page only work in an anconda environment. Therefore, first download and install [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html). Then create an environment called pcsmac with following command
```
conda create -n pcsmac python=3.6
```

### Unix

Install SMAC3 with pipeline caching integration from command line. Install SWIG for the C++ random forest.
```
conda install swig
git clone https://github.com/jtuyls/SMAC3_4_PC.git
cd SMAC3_4_PC
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install
```
Clone repository and install requirements.
```
pip install -r requirements.txt
```

### MAC OS X

For installation on Mac OS X there is a problem with installing SMAC3 as above. SMAC is dependenct on pyrfr that uses the gcc compiler for c++ while Mac OS X uses the clang compiler. Therefore we install a gcc compiler within anaconda. Furthermore, install SWIG for the C++ random forest.

Install SMAC3 with pipeline caching integration from commandline in anaconda environment. Therefore first install pyrfr separately with anaconda gcc compiler.
```
conda install swig
conda install gcc
CC=/Users/[username]/anaconda/bin/gcc pip install pyrfr --no-cache-dir
git clone https://github.com/jtuyls/SMAC3_4_PC.git
cd SMAC3_4_PC
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install
```
Clone repository and install requirements.
```
pip install -r requirements.txt
```

## Optimization of machine learning pipeline

First go to the directory above pc_smac. Then the optimization can be started using following generic command:

```
python -m pc_smac.run -a=[acquistion function, STRING: e.g pc-m-pceips] 
                      -di=[double intensification enabled: INT: 1 (yes) 0 (no)] 
                      -w=[wallclock time in seconds, INT: e.g. 1800]
                      -l=[Location of the dataset: STRING: DEFAULT: if not provided, data/46_bac is used as example]
                      -m=[OPTIONAL: memory limit in Mb: INT: default=6000]
                      -r=[OPTIONAL: maximum number of runs: INT: DEFAULT=10000] 
                      -c=[OPTIONAL: cutoff for each evaluation in seconds: INT: DEFAULT=1800] 
                      -s=[OPTIONAL: stamp used for identifying the directory and names of the results: STRING: DEFAULT=stamp] 
                      -o=[OPTIONAL: output directory: STRING: DEFAULT=pc_smac/output]
                      -cd=[OPTIONAL: cache directory: STRING: DEFAULT=pc_smac/cache]
                      -ps=[OPTIONAL: pipeline space used: STRING: DEFAULT=None; Example of small space: nystroem_sampler-sgd]
                      -ifs=[OPTIONAL: intensification fold size: INT: DEFAULT=10]
                      -sn=[OPTIONAL: splitting number ! only for sigmoid and mrs random search version: INT: DEFAULT=5]
                      -rs=[OPTIONAL: random splitting enabled ! only for mrs random search version: INT: DEFAULT=0 (no)]
```

### Example

Execute following command to test optimization with pipeline caching (and marginalization) for 1800 seconds or 100 evaluations on [OpenML dataset 46](https://www.openml.org/d/46).

```
python -m pc_smac.run -a=pc-m-pceips -di=0 -w=1800 -r=100 -c=100 -m=4000 -s=pc-m-pceips
```

Don't forget to clean caches afterwards because they can take quite some space!


