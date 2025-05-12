import os

def delete_files_with_keyword(folder_path, keyword):
    deleted_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if keyword in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                deleted_files.append(file_path)
    return deleted_files

# Gán tham số cần xóa (ví dụ: "aug", "lr", "batch", ...)
tuning_hyperparameters = "lr"

# Thư mục cần kiểm tra
folders = ["densenet121/logs", "densenet121/models"]

all_deleted = []
for folder in folders:
    deleted = delete_files_with_keyword(folder, tuning_hyperparameters)
    all_deleted.extend(deleted)

# In kết quả
print(f"✅ Deleted {len(all_deleted)} files containing '{tuning_hyperparameters}':")
for f in all_deleted:
    print(f" - {f}")
