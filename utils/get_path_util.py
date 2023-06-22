import os


def get_path(folder, file_name):
    current_dir = os.getcwd()
    relative_path = folder
    folder_path = os.path.join(current_dir, relative_path)
    full_path = f'{folder_path}/{file_name}'
    return full_path
