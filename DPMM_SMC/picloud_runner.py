
import sys

import cloud
cloud.setkey(7513, api_secretkey='ca43a3535fa17e28b687f0f1691c67db261392ae')
cloud_environment = 'Julia'

"""
number_of_clusters = int(sys.argv[1])
if_zero_shortlearning = sys.argv[2] # Should be "yes" or "no"
experiment_name = sys.argv[3]"""
TRIALS = int(sys.argv[1])


def run_on_instance(trial_id):
  global number_of_clusters
  global if_zero_shortlearning
  global experiment_name
  import subprocess
  import os
  os.environ['DISPLAY'] = ":1"
  print "Starting"
  ls_output = subprocess.Popen(["/Users/tejas/Documents/julia/julia", "runner.jl", str(trial_id)], \
                               cwd = "/Users/tejas/Documents/MIT/Samplers/DPMixtureModel/DPMM_SMC/") #, \
                               # stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, err = ls_output.communicate()
  print "Finished"
  return out
 
#result = run_on_instance(1)  

jids = cloud.map(run_on_instance, range(TRIALS), _env=cloud_environment, _type='c1', _cores=1)
print jids
result = cloud.result(jids)

print result