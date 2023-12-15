from init import add_hpc_to_sys_path
add_hpc_to_sys_path()
from config.SSH_client import SSHClient
from config.config import get_config

# Initiate SSH client
config = get_config()['lumi']
ssh_client = SSHClient(config)
ssh_client.connect()

# Uploading job file
remote_job_file = 'job.sh'
local_job_file = f'{config["projectpath"]}HPC/LUMI/jobs/{remote_job_file}'
ssh_client.upload_file(local_job_file, remote_job_file)

# Uploading model file
remote_model_file = 'mainDANSUM_lumi.py'
local_model_file = f'{config["projectpath"]}model/{remote_model_file}'
ssh_client.upload_file(local_model_file, remote_model_file)

# Submitting job and waiting for completion
job_id = ssh_client.submit_job(remote_job_file)
ssh_client.wait_for_job_completion(job_id)

# Retrieving output file
ssh_client.show_output(job_id)

# Closing connection
ssh_client.close_connection()
