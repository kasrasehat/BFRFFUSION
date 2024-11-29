import os
import random
import shutil

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Step 1: List all files in the source directory
    all_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    total_files = len(all_files)
    if total_files == 0:
        print("No files found in the source directory.")
        return

    # Step 2: Randomly shuffle the list of files
    random.shuffle(all_files)

    # Step 3: Calculate the number of files for each set
    train_count = int(total_files * train_ratio)
    val_count = int(((total_files * val_ratio)//32)*32)
    test_count = total_files - train_count - val_count  # To ensure all files are used

    # Step 4: Split the files into train, validation, and test sets
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]

    # Step 5: Copy the files to the respective directories
    datasets = [
        (train_files, train_dir, 'Training'),
        (val_files, val_dir, 'Validation'),
        (test_files, test_dir, 'Test')
    ]

    for files_list, dest_dir, dataset_name in datasets:
        print(f"Copying {len(files_list)} files to {dataset_name} directory: {dest_dir}")
        os.makedirs(dest_dir, exist_ok=True)
        for file_path in files_list:
            # Copy files to the destination directory
            dest_path = os.path.join(dest_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
        print(f"Finished copying files to {dataset_name} directory.")

    print("Dataset splitting completed successfully.")

# Example usage
if __name__ == "__main__":
    source_directory = "/home/user1/kasra/pycharm-projects/BFRffusion/assets/balance_all_data/"
    train_directory = "/home/user1/kasra/pycharm-projects/BFRffusion/assets/train/hq"
    validation_directory = "/home/user1/kasra/pycharm-projects/BFRffusion/assets/validation/hq"
    test_directory = "/home/user1/kasra/pycharm-projects/BFRffusion/assets/test/hq"

    # Set the desired proportions
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    split_dataset(source_directory, train_directory, validation_directory, test_directory,
                  train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
