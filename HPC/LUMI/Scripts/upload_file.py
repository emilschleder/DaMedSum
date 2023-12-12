from init import add_hpc_to_sys_path
add_hpc_to_sys_path()
from config.SSH_client import SSHClient
from config.config import get_config

# Initiate SSH client
ssh_client = SSHClient(get_config()['lumi'])
ssh_client.connect()

# Uploading job file 
local_path = '/Users/emilschledermann/git/ResearchProject/HPC/LUMI/jobs/job.sh'
ssh_client.upload_file(local_path, local_path.split('/')[-1])

# Closing connection
ssh_client.close_connection()