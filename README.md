# HAMMER_on_SMARTS

## Usage

```bash
virtualenv VENV
source VENV/bin/activate 
pip install --upgrade pip 
pip install -r requirements.txt 
```

## Running experiments on PettingZoo Environments ---

```python
python hammer-run.py --envname ENV --config CONFIG 
[-h HELP] 
[--expname EXPNAME] [--nagents NAGENTS] 
[--sharedparams SHAREDPARAMS] [--maxepisodes MAXEPISODES] 
[--partialobs PARTIALOBS] [--heterogeneity HETEROGENEITY] 
[--limit LIMIT] [--maxcycles MAXCYCLES] 
[--dru_toggle DRU_TOGGLE] [--meslen MESLEN] 
[--randomseed RANDOMSEED] [--saveinterval SAVEINTERVAL] 
[--logdir LOGDIR] [--savedir SAVEDIR]
```

### Cooperative Navigation (MPE) and MultiWalker (SISL)

Use

`ENV="cn"`, `CONFIG="config/cn.yaml"` for Cooperative Navigation, and

`ENV="mw"`, `CONFIG="config/mw.yaml"` for MultiWalker.

*Currently HAMMER supports only CN and MW, but other environments are easy to add in the code. We will upgrade the framework to flexibly switch among pettinzoo environments in the future.

## SMARTS

Setup:

```bash
git clone https://github.com/huawei-noah/SMARTS.git
cd SMARTS
```
Follow the instructions given by prompt for setting up the SUMO_HOME environment variable
```bash
./install_deps.sh
```
enter to your bashrc file and add SUMO_HOME to the end of the file
```
nano .bashrc
export SUMO_HOME=/usr/share/sumo/
```

verify sumo is >= 1.5.0

if you have issues see ./doc/SUMO_TROUBLESHOOTING.md
```
sumo
```

setup virtual environment; presently only Python 3.7.x is officially supported
```bash
python3.7 -m venv .venv
```

enter virtual environment to install all dependencies
```bash
source .venv/bin/activate
```

upgrade pip, a recent version of pip is needed for the version of tensorflow we depend on
```bash
pip install --upgrade pip
```

install [train] version of python package with the rllib dependencies
```bash
pip install -e .[train]
```

make sure you can run sanity-test (and verify they are passing)

if tests fail, check './sanity_test_result.xml' for test report. 
```bash
pip install -e .[test]
make sanity-test
```

### Building a scenario in SMARTS:

1. open *supervisord.conf* and edit line no. 3

For example, to open a sceanario consisting of 4 lanes: 
> command=python examples/single_agent.py scenarios/intersections/4lane

2. building the newly changed scenario, replace the last path by the path of scenario you want to configure:
> scl scenario build --clean scenarios/intersections/4lane

3. simualation and visualization:
>  supervisord
