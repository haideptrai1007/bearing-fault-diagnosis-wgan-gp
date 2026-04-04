import os
import scipy.io

def cwru_read_mat(root_path):
    data = {}
    if not os.path.isdir(root_path):
        raise ValueError(f"{root_path} is not a valid directory")
    for fault_type in os.listdir(root_path):
        fault_path = os.path.join(root_path, fault_type)
        if not os.path.isdir(fault_path):
            continue
        data[fault_type] = {}
        for fault_size in os.listdir(fault_path):
            size_path = os.path.join(fault_path, fault_size)
            if not os.path.isdir(size_path):
                continue
            data[fault_type][fault_size] = {}
            for item in os.listdir(size_path):
                item_path = os.path.join(size_path, item)

                if os.path.isdir(item_path):
                    mat_list = []
                    for f in os.listdir(item_path):
                        if f.endswith(".mat"):
                            mat_list.append(
                                scipy.io.loadmat(os.path.join(item_path, f))
                            )
                    if mat_list:
                        data[fault_type][fault_size][item] = mat_list
                elif item.endswith(".mat"):
                    data[fault_type][fault_size].setdefault("default", [])
                    data[fault_type][fault_size]["default"].append(
                        scipy.io.loadmat(item_path)
                    )
    return data


def cwru_read_normal(root_path):
    data = {}
    if not os.path.isdir(root_path):
        raise ValueError(f"{root_path} is not a valid directory")
    for file_name in os.listdir(root_path):
        file_path = os.path.join(root_path, file_name)
        if file_name.endswith(".mat") and os.path.isfile(file_path):
            key = os.path.splitext(file_name)[0]
            data[key] = scipy.io.loadmat(file_path)
    return data