import os
import random
import shutil

def make_validation(train_dir="dataset/train", val_dir="dataset/val", val_split=0.1):
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    else:
        return "Validation directory already exists."

    all_files = os.listdir(train_dir)
    num_val_files = int(len(all_files) * val_split)

    val_files = random.sample(all_files, num_val_files)

    for file_name in val_files:
        train_file_path = os.path.join(train_dir, file_name)
        val_file_path = os.path.join(val_dir, file_name)

        shutil.move(train_file_path, val_file_path)

    print(f"Moved {num_val_files} files from {train_dir} to {val_dir}.")

if __name__ == "__main__":
    make_validation()