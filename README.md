# pc_prototype

## Intallation

Use python 3.5.

### Unix

Install SMAC3 from command line.
```
pip clone https://github.com/automl/SMAC3.git
cd SMAC3
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install
```
Install other requirements.
```
pip install requirements.txt
```

### MAC OS X

For installation on Mac OS X there is a problem with installing SMAC3 as above. SMAC is dependenct on pyrfr that uses the gcc compiler for c++ while Mac OS X uses the clang compiler. Therefore use an anaconda environment and install a gcc compiler within anaconda.

Install SMAC3 from commandline in anaconda environment. Therefore first install pyrfr separately with anaconda gcc compiler.
```
conda install gcc
CC=/Users/username/anaconda/bin/gcc pip install pyrfr --no-cache-dir
pip clone https://github.com/automl/SMAC3.git
cd SMAC3
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install
```
Install other requirements.
```
pip install requirements.txt
```
