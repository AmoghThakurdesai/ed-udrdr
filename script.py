import os
import shutil

# Define the parent directory
parent_dir = '/mnt/d/PythonFiles/assignment/dataset/GT-RAIN_train'

# Iterate over all the subdirectories in the parent directory
for subdir in os.listdir(parent_dir):
    subdir_path = os.path.join(parent_dir, subdir)

    # Check if it's a directory
    if os.path.isdir(subdir_path):
        # Iterate over all the files in the subdirectory
        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)

            # Move the file to the parent directory
            shutil.move(file_path, parent_dir)

print("All files have been moved to the parent directory.")
