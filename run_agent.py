import os
import signal
import sys
import argparse
import subprocess
from time import sleep
from datetime import datetime

_print = print


# monkey patch print statement to show timestamps
def print(*args, **kwargs):
    if len(args) == 0:
        return _print(*args, **kwargs)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return _print(f'{ts} |', *args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start an experiment')

    parser.add_argument('--sweep_id', '-s', type=str, help='The W&B sweep id used to start the agents.')
    parser.add_argument('--gpus', '-g', type=str, help='Which GPUs to use for running agents on. '
                                                       'This will internally set the CUDA_VISIBLE_DEVICES environment'
                                                       'variable.')
    parser.add_argument('--n_parallel', '-p', type=int, default=1, required=False,
                        help='The number of agents to run in parallel on each GPU')

    args = parser.parse_args()

    sweep_id = args.sweep_id
    gpus_str = args.gpus
    gpus = [int(g) for g in gpus_str.strip().split(',')]
    n_parallel = args.n_parallel

    verbose = (n_parallel * len(gpus)) == 1
    print('=' * 100)
    if verbose:
        print('Note: Showing all stdout and stderr output, as only a single agent is running!')
    else:
        print('Note: Catching all stdout and stderr output as multiple agents run in parallel!')
    print('=' * 100)
    print()

    processes = []
    # start all processes
    for i in range(n_parallel):
        for g in gpus:
            command = f'CUDA_VISIBLE_DEVICES={g} wandb agent {sweep_id}'
            print(f'Launching command "{command}" as {i}. process on gpu {g}')

            kwargs = {}
            if not verbose:
                # hide output if not verbose
                kwargs = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
            p = subprocess.Popen(command, preexec_fn=os.setsid, shell=True, **kwargs)

            # keep reference to Popen object
            processes.append((p, (i, g)))

    print()
    print('All processes launched')
    print()

    try:
        # wait for completion of all processes
        while len(processes) > 0:
            finished_processes = []
            # continuously check which processes are running
            for pi in processes:
                p, (i, g) = pi
                # check whether process finished
                if p.poll() is not None:
                    print(f'{i}. process on gpu {g} finished')
                    finished_processes.append(pi)

            # remove all finished processes from collection
            for fi in finished_processes:
                processes.remove(fi)

            # wait some time before checking again
            sleep(1)

        print()
        print('All processes finished')
        print()

    except (KeyboardInterrupt, SystemExit):
        print()
        print('Terminating all processes...')
        print()

        # repeat until all processes are killed
        while running_processes := [p for p, _ in processes if p.poll() is None]:
            # forcefully terminate processes if necessary
            for p in running_processes:
                # kill process
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                os.kill(p.pid, signal.SIGTERM)
                print('killing process', p.pid)
            sleep(1)
        sys.exit()
