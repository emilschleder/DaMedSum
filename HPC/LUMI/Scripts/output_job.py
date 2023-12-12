from init import add_hpc_to_sys_path, add_config_to_sys_path
add_hpc_to_sys_path()
add_config_to_sys_path()
from HPC_interaction_module.SSH_client import SSHClient
from Config.config import get_config

# Initiate SSH client
config = get_config()['lumi']
ssh_client = SSHClient(config)
ssh_client.connect()

job_ids = [5084835, 5082380, 5069383]

# Retrieving output file
for job in job_ids:
    ssh_client.show_output(job_id=job)

# Closing connection
ssh_client.close_connection()