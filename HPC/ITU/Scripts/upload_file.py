from HPC.config.SSH_client import SSHClient

# Initiate SSH client
ssh_client = SSHClient(host='hpc.itu.dk', username='easc')
ssh_client.connect()

# Uploading job file 
remote_job_file = 'summaries.csv'
local_path = f'/Users/emilschledermann/git/ResearchProject/data/{remote_job_file}'
ssh_client.upload_file(local_path, remote_job_file)

# Closing connection
ssh_client.close_connection()