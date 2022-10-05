import os, yaml 
from pathlib import Path 
from itertools import count 

dumpdir = "runs_new_cn_modified/" 
if not os.path.isdir(dumpdir):
    os.mkdir(dumpdir)
fixed_text = "#!/bin/bash\n"\
             "#SBATCH --nodes=1\n"\
             "#SBATCH --cpus-per-task=16 \n"\
             "#SBATCH --time=4:00:00\n"\
             "#SBATCH --mem=40GB\n"

config_file = "configs/cn.yaml"

with open(config_file, "r") as stream:
        try: config = yaml.safe_load(stream) 
        except yaml.YAMLError as exc: print(exc) 



for n_agents in [3, 5, 7, 10]: 
    for dru_toggle in [0, 1]: 
        for meslen in [0, 1, 2]: 
            for randomseed in [ 2528, 66962, 34046, 58876, 84054, 34131, 33989, 59004, 94644, 98216 ]: 
                exp = str(randomseed) 
                command = fixed_text + "#SBATCH --job-name="+exp+"\n#SBATCH --output="+exp+".out\n"
                command += "\nsource ../venvs/hammer/bin/activate\n"\
                    "\nmodule load python/intel/3.8.6\n"\
                    "module load openmpi/intel/4.0.5\n"\
                    "time python3 hammer-run.py " 
                command = ' '.join([
                    command, 
                    '--envname cn', 
                    '--config configs/cn.yaml', 
                    '--nagents', str(n_agents),  
                    '--dru_toggle', str(dru_toggle), 
                    '--meslen', str(meslen), 
                    '--partialobs 1', 
                    '--heterogeneity 0', 
                    '--randomseed', str(randomseed), 
                    ]) 
                # print(command) 
                log_dir = Path(dumpdir)
                for i in count(1):
                    temp = log_dir/('run{}.sh'.format(i)) 
                    if temp.exists():
                        pass
                    else:
                        with open(temp, "w") as f:
                            f.write(command) 
                        log_dir = temp
                        break 
