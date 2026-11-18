"""PEAR simulator process with post-exec parent ownership setup."""
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess

from pear.sim.process import SimulatorProcess


class SpawnedSimulatorProcess(SimulatorProcess):
    def start(self):
        conn,process_end=Pipe();self.start_n+=1
        self.log_path=self.log_dir/f'{self.name}-{self.start_n}.log'
        if self.command[1:3]!=['-u','-m']:raise ValueError('Unexpected PEAR simulator command')
        bootstrap=Path(__file__).with_name('efe_pear_child.py')
        command=[*self.command[:2],str(bootstrap),str(os.getpid()),*self.command[3:],
            '--fd',str(process_end.fileno()),'--',*self.context_args]
        try:
            with self.log_path.open('ab')as log:
                self.popen=subprocess.Popen(command,cwd=self.cwd,env=self.variables,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    pass_fds=(process_end.fileno(),),start_new_session=True)
        except BaseException:
            conn.close();raise
        finally:process_end.close()
        self.conn=conn
