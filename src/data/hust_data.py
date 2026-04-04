import os


def hust_load_data(root_path):
    files, labels = [], []
    for file in os.listdir(root_path):
        file_path = os.path.join(root_path, file)
        files.append(file_path)
        if len(file) < 9:
            if file[0] == "B":
                labels.append("B")
            elif file[0] == "I":
                labels.append("IR")
            elif file[0] == "O":
                labels.append("OR")
            elif file[0] == "N":
                labels.append("N")
            else:
                continue
    return files, labels