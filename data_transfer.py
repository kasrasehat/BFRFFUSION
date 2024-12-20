import os
import random
import shutil

def copy_random_files(source_dir, dest_dir, num_files, preserve_structure=False):
    # Step 1: List all files in the source directory
    all_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    # Check if the number of files is less than requested
    total_files = len(all_files)
    if total_files == 0:
        print("No files found in the source directory.")
        return
    elif total_files < num_files:
        print(f"Only {total_files} files found. Proceeding to copy all of them.")
        num_files = total_files

    # Step 2: Randomly select the specified number of files
    selected_files = random.sample(all_files, num_files)

    # Step 3: Copy the selected files to the destination directory
    for file_path in selected_files:
        if preserve_structure:
            # Preserve directory structure
            relative_path = os.path.relpath(file_path, source_dir)
            dest_path = os.path.join(dest_dir, relative_path)
            dest_folder = os.path.dirname(dest_path)
            os.makedirs(dest_folder, exist_ok=True)
        else:
            # Place all files directly into the destination directory
            dest_path = os.path.join(dest_dir, os.path.basename(file_path))
        
        shutil.copy2(file_path, dest_path)
        print(f"Copied {file_path} to {dest_path}")

    print(f"Successfully copied {num_files} files to {dest_dir}")

# Example usage
if __name__ == "__main__":
    source_directory = "/mnt/drive/datasets/datasets/DF2K/DF2K_HR/"
    destination_directory = "/home/user1/kasra/pycharm-projects/BFRffusion/assets/balance_all_data/"
    number_of_files_to_copy = 4000

    copy_random_files(source_directory, destination_directory, number_of_files_to_copy, preserve_structure=False)
