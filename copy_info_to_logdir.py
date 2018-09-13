import sh
from sh import git, cp
from os import path

def get_commit_id():
  return git('rev-parse', 'HEAD').stdout#.decode('utf-8')

def get_commit_message():
  return git('log', '-1').stdout#.decode('utf-8')

def write_to_file(filepath, contents):
  filehandle = open(filepath, 'wb')
  filehandle.write(contents)
  return

def copy_info_to_logdir(logdir):
  write_to_file(path.join(logdir, 'commit_id.txt'), get_commit_id())
  write_to_file(path.join(logdir, 'commit_message.txt'), get_commit_message())
  cp('train_egan_celeba.py', path.join(logdir, 'train_egan_celeba.py'))
  cp('models/models_egan_celeba.py', path.join(logdir, 'models_egan_celeba.py'))
