import os, yaml 
from pathlib import Path 
from itertools import count 

dumpdir = "runs/" 
if not os.path.isdir(dumpdir):
    os.mkdir(dumpdir)
fixed_text = "#!/bin/bash\n"\
             "#SBATCH --nodes=1\n"\
             "#SBATCH --cpus-per-task=16 \n"\
             "#SBATCH --time=48:00:00\n"\
             "#SBATCH --mem=40GB\n"

config_file = "configs/mw.yaml"

with open(config_file, "r") as stream:
        try: config = yaml.safe_load(stream) 
        except yaml.YAMLError as exc: print(exc) 



# for dru_toggle in [0, 1]: 
#     for meslen in [1,2,3]: 
for dru_toggle in [0]: 
    for meslen in [0]: 
        for randomseed in [14712, 10453, 92959, 61033, 90300]: 
            exp = str(randomseed) 
            command = fixed_text + "#SBATCH --job-name="+exp+"\n#SBATCH --output="+exp+".out\n"
            command += "\nsource ../venvs/hammer/bin/activate\n"\
                "\nmodule load python/intel/3.8.6\n"\
                "module load openmpi/intel/4.0.5\n"\
                "time python3 hammer-run.py " 
            command = ' '.join([
                command, 
                '--envname mw', 
                '--config configs/mw.yaml', 
                '--nagents 3', 
                '--dru_toggle', str(dru_toggle), 
                '--meslen', str(meslen), 
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