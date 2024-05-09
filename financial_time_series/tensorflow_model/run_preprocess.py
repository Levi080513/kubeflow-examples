"""Module for running the data retrieval and preprocessing.

Scripts that performs all the steps to get the train and perform preprocessing.
"""
import logging
import argparse
import sys
import shutil
import os

#pylint: disable=no-name-in-module
from helpers import preprocess
from helpers import storage as storage_helper


def parse_arguments(argv):
  """Parse command line arguments
  Args:
      argv (list): list of command line arguments including program name
  Returns:
      The parsed arguments as returned by argparse.ArgumentParser
  """
  parser = argparse.ArgumentParser(description='Preprocessing')


  parser.add_argument('--start_date',
                      type=str,
                      help='111',
                      default='2010-10-01')
  parser.add_argument('--end_date',
                      type=str,
                      help='222',
                      default='2022-10-01')

  parser.add_argument('--kfp',
                      dest='kfp',
                      action='store_true',
                      help='Kubeflow pipelines flag')

  args, _ = parser.parse_known_args(args=argv[1:])

  return args


def run_preprocess(argv=None):
  """Runs the retrieval and preprocessing of the data.

  Args:
    args: args that are passed when submitting the training

  Returns:

  """
  logging.info('starting preprocessing of data..')
  args = parse_arguments(sys.argv if argv is None else argv)
  closing_data = preprocess.load_data(args.start_date, args.end_date)
  time_series = preprocess.preprocess_data(closing_data)
  logging.info('preprocessing of data complete..')

  logging.info('starting save data to pv')
  temp_folder = '/data'
  if not os.path.exists(temp_folder):
    os.mkdir(temp_folder)
  file_path = os.path.join(temp_folder, 'data_{}_{}.csv'.format(args.start_date, args.end_date))
  time_series.to_csv(file_path, index=False)
  logging.info('save data to pv completed..')


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  run_preprocess()
