"""Add failure evidence around native validation, preserving every native guard."""
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    import gr00t_sim.embodiments.common.grippers as grippers
    from gr00t_sim.repo_paths import resolve_repo_path
    original=grippers.validate_gripper_configuration

    def checked(config):
        try:return original(config)
        except Exception:
            rows=[]
            for name,robot,definition in grippers._robots(config):
                def evidence(path):
                    p=Path(path)
                    return {'path':str(p),'resolved':str(p.resolve()),'exists':p.is_file(),
                        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()if p.is_file()else None}
                rows.append({'arm':name,'declared_gripper':definition.name,
                    'actual':evidence(robot.spawn.usd_path),'expected':evidence(resolve_repo_path(definition.geometry))})
            print('PEAR_NATIVE_GRIPPER_DIAGNOSTIC '+json.dumps(rows),flush=True)
            raise
    grippers.validate_gripper_configuration=checked
    source=Path(sys.argv[1]);sys.argv=[str(source),*sys.argv[2:]]
    runpy.run_path(str(source),run_name='__main__')


if __name__=='__main__':main()
