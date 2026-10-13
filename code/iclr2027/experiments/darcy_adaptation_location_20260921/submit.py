"""Submit one auditable dependency chain; never duplicate a submitted campaign."""
import json
import subprocess
from experiment import HERE,ROOT,OUT,PACKAGE,sha,save,read,verify_protocol

verify_protocol()
assert read(PACKAGE/'preflight.json')['passed']
assert not (HERE/'jobs.json').exists()
paths=[HERE/'experiment.py',HERE/'test_experiment.py',HERE/'run_stage.sbatch',HERE/'submit.py']
save(PACKAGE/'implementation_freeze.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in paths},protocol_sha256=sha(PACKAGE/'protocol.json')))
(OUT/'slurm').mkdir(exist_ok=True)
jobs={}
def submit(name,phase,dependency=None,array=None):
    command=['sbatch','--parsable','--job-name=darcy-location-'+name,
        '--output='+str(OUT/'slurm'/(name+'_%A_%a.out')),
        '--error='+str(OUT/'slurm'/(name+'_%A_%a.err'))]
    if dependency:command+=['--dependency='+dependency]
    if array:command+=['--array='+array]
    command += [str(HERE/'run_stage.sbatch'),phase]
    result=subprocess.run(command,check=True,capture_output=True,text=True)
    job=result.stdout.strip().split(';')[0]
    jobs[name]={'job_id':job,'command':command};save(HERE/'jobs.json',jobs)
    print(name,job,flush=True);return job
screen=submit('screen','screen',array='0-5%4')
selection=submit('select','select','afterok:'+screen)
training=submit('train','train','afterok:'+selection,array='0-11%4')
freeze=submit('freeze','freeze-test','afterok:'+training)
tests=submit('test','test','afterok:'+freeze,array='0-13%4')
submit('collect','collect','afterany:'+tests)
# Preserve failure records even when failed jobs block later phases.
submit('screen-ledger','collect','afterany:'+screen)
submit('training-ledger','collect','afterany:'+training)
