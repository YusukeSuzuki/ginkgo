import argparse
from datetime import datetime
from os.path import expandvars
from pathlib import Path
import os
import selectors
import signal
from subprocess import Popen, PIPE
import sys
from time import sleep

import yaml

def get_argument_parser():
    parser = argparse.ArgumentParser(description='ginkgo batch learning program')
    parser.add_argument(
        'config_yaml', nargs=1, type=argparse.FileType('r'),
        help='configuration yaml file')
    parser.add_argument('--verbose', '-v', action='store_true')

    return parser

def datetime_directory_name(dt):
    return dt.strftime("%Y%m%d_%H%M%S.%f")

def print_if(cond, *msg):
    if cond:
        print(msg)

def get_path(path):
    return Path(expandvars(path))

def run_batches(config, verbose):
    outdir = get_path(config['settings'].get('output_directory', '.'))

    if outdir.is_file():
        print('ERROR: output place ({}) is not directory'.format(outdir))
        exit(1)

    old_results = {}

    outdir.mkdir(parents=True, exist_ok=True)
    outdir = outdir.resolve()
    os.chdir(str(outdir))

    for d in outdir.iterdir():
        finish_yaml = d/'finish.yaml'

        if finish_yaml.is_file():
            print_if(verbose, 'old result is found at {}'.format(d))
            old_results[d] = yaml.load(finish_yaml.open('r'))

    if not config.get('batches', None):
        print('no batches in configuration yaml file'. format())

    for batch in config['batches']:
        print('--------------------')
        found = False

        for old_path, old_result in old_results.items():
            if batch['name'] == old_result['name']:
                found = True
                print('batch "{}" has be done at {}, skip'.format(batch['name'], old_path))
                break

        if found:
            continue

        batch_out_dir = Path(datetime_directory_name(datetime.now()))
        batch_out_dir.mkdir()
        os.chdir(str(batch_out_dir))

        print_if(verbose, batch)

        # do batch
        if not batch.get('commands'):
            print('there is no "commands" field in batch entry')
            break

        returncode = 0

        for command in batch['commands']:
            command_line = [command['command']] + command['args']
            print("run command line: {}".format(" ".join(command_line)))

            try:
                proc = Popen(command_line, bufsize=1,
                    universal_newlines=True, stdout=PIPE, stderr=PIPE)

                tee_out_proc = Popen(['tee', '-a', 'stdout.txt'],
                    bufsize=1, universal_newlines=True, stdin=proc.stdout, stdout=sys.stdout)
                tee_err_proc = Popen(['tee', '-a', 'stderr.txt'],
                    bufsize=1, universal_newlines=True, stdin=proc.stderr, stdout=sys.stderr)

                tee_out_proc.wait()
                tee_err_proc.wait()
                proc.wait()
                
                if proc.returncode != 0:
                    returncode = proc.returncode
                    break
            except KeyboardInterrupt as e:
                for p in [proc, tee_out_proc, tee_err_proc]:
                    if p is not None:
                        p.kill()
                raise e

        if returncode != 0:
            batch['finished_time'] = datetime.now()
            yaml.dump(batch, Path('failure.yaml').open('w'), default_flow_style=False)
            print('batch failed: {}'.format(batch['name']))
            break

        os.chdir(str(outdir))
        batch_out_dir.rename(batch['name'])
        batch_out_dir = Path(batch['name'])

        batch['finished_time'] = datetime.now()
        yaml.dump(batch, (batch_out_dir/'finish.yaml').open('w'), default_flow_style=False)
        print('batch done: {}'.format(batch['name']))

        sleep(0.01)

def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    print('reading configuration file ({})'.format(args.config_yaml[0].name))
    config = yaml.load(args.config_yaml[0])

    run_batches(config, args.verbose)

if __name__ == '__main__':
    main()

