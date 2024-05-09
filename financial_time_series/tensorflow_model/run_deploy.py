"""Module for deploying a machine learning model to TF serving.

Scripts that performs the steps to deploy a model with TF serving
"""
import argparse
import logging
import sys
import subprocess


def parse_arguments(argv):
  """Parse command line arguments
  Args:
      argv (list): list of command line arguments including program name
  Returns:
      The parsed arguments as returned by argparse.ArgumentParser
  """
  parser = argparse.ArgumentParser(description='Preprocessing')


  parser.add_argument('--tag',
                      type=str,
                      help='tag of the model',
                      required=True)

  args, _ = parser.parse_known_args(args=argv[1:])

  return args


def run_deploy(argv=None):
  """Runs the retrieval and preprocessing of the data.

  Args:
    args: args that are passed when submitting the training

  """
  args = parse_arguments(sys.argv if argv is None else argv)
  logging.info('start deploying model %s ..', args.tag)


  # copy the files
  logging.info('deploying model %s on TF serving', args.tag)
  src_folder = '/models/{}'.format( args.tag)
  target_folder = '/serving/{}'.format(args.tag)
  subprocess.call(['cp', '-r', src_folder, target_folder])


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  run_deploy()
