import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.SSH_client import SSHClient
from config.config import get_config

# Initiate SSH client
ssh_client = SSHClient(get_config()['itu'])
ssh_client.connect()


job_ids = [5053728]

# Retrieving output file
for job in job_ids:
    ssh_client.show_output(hpc_prefix=get_config()['itu']['job_prefix'], job_id=job)

# Closing connection
ssh_client.close_connection()