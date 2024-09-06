# THIS IS A CONFIG.PY EXAMPLE, RENAME THIS TO "config.py" AND PUT THE PATH YOU WANT TO USE

user_folder_path = "./"

# Google Colab works better for Windows users, 
# pySUS is not working on Windows for me, had to use WSL
user_folder_path_to_save = '/content/drive/My Drive/'

def get_user_folder_path():
    return user_folder_path

def get_user_folder_path_to_save():
    return user_folder_path_to_save