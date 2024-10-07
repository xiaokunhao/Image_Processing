from DataInit.file_utils import create_directory

folder_path = 'my_folder'
result = create_directory(folder_path)
if result:
    print(f"The folder '{folder_path}' already exists.")
else:
    print(f"The folder '{folder_path}' was successfully created.")