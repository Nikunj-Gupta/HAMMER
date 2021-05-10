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

### SMARTS

WIP
