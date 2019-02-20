import os

def make_dirs(*dir_names):
    '''
    创建多个目录
    '''
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)