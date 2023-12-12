import paramiko
import getpass
import time
import os
from scp import SCPClient

class SSHClient:
    def __init__(self, config: dict):
        self.host = config['hostname']
        self.username = config['username']
        self.password = config['password']
        self.workdir = config['workdir']
        self.hpc_prefix = config['job_prefix']
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(f'Initialized SSH client for {self.username}@{self.host}')

    def connect(self):
        if self.password is None:
            self.password = getpass.getpass(f'Enter password for {self.username}@{self.host}: ')
        print(f'connecting to {self.host} as {self.username}')
        self.ssh.connect(self.host, username=self.username, password=self.password)
        
    def upload_file(self, local_path: str, remote_path: str):
        with SCPClient(self.ssh.get_transport()) as scp:
            dir = f'{self.workdir}{remote_path}'
            print(f'Uploading {local_path} to {dir}')
            scp.put(local_path, dir)

    def submit_job(self, job_script: str):
        if self.workdir != '':
            cmd = f'cd {self.workdir} && sbatch {job_script}'
        else:
            cmd = f'sbatch {job_script}'
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        job_id = int(stdout.read().decode().split()[-1])
        print(f'Submitted job with ID {job_id}')
        return job_id
    
    def cancel_job(self, job_id: int):
        self.ssh.exec_command(f'scancel {job_id}')
        print(f'Cancelled job with ID {job_id}')

    def wait_for_job_completion(self, job_id: int):
        job_finished = False
        while not job_finished:
            time.sleep(30)
            stdin, stdout, stderr = self.ssh.exec_command(f'sacct -j {job_id} --format=State --noheader')
            output = stdout.read().decode().strip().split('\n')
            for line in output:
                if 'COMPLETED' in line:
                    job_finished = True
                    print(f'Job {job_id} has completed')
                    break
                elif 'RUNNING' in line:
                    print(f'Job {job_id} is still running...')
                    break

    def show_output(self, job_id: int):
        job_output_filename = f'{self.workdir}{self.hpc_prefix}{job_id}.out'
        local_output_path = os.path.join(os.getcwd(), f'{self.hpc_prefix}{job_id}.out')
        
        with SCPClient(self.ssh.get_transport()) as scp:
            scp.get(job_output_filename, local_output_path)
        
        print(f'Downloaded .out file to {local_output_path}')
        
        with open(local_output_path, encoding='utf-8', errors='replace') as file:
            for line in file:
                print(line)

    def close_connection(self):
        self.ssh.close()