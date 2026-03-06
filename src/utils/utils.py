import numpy as np
from scipy.signal import resample_poly
import scipy.io
from sklearn.model_selection import train_test_split
import torch

def sliding_window(signal, window_size=2048, overlap=0.75):
    stride = int(window_size * (1 - overlap))
    windows = []
    for i in range(0, len(signal) - window_size + 1, stride):
        window = signal[i:i + window_size]
        windows.append(window)
    return np.array(windows)

def time_frequency_transform(file, label, type, transform, img_size=128, normal=False, gray=False):
    mat_file = scipy.io.loadmat(file)
    images = []
    labels = []
    for i, j in mat_file.items():
        if type in i:
            signal = j.squeeze()
            if normal:
                signal = resample_poly(signal, up=1, down=4)

            windows = sliding_window(signal, 2048, 0.75)
            for window in windows:
                img = transform(window, img_size=img_size, gray=gray)
                images.append(img)
                labels.append(label)
    
    label_map = {"Normal": 0, "OR": 1, "IR": 2, "B": 3}
    labels = np.asarray([label_map[label] for label in labels])

    return np.array(images), labels

def train_test_val_by_load(load_data, path, type, trans, img_size=128, normal=False, gray=False):
    files, labels = load_data(path, normal)

    train = {'data': [], 'label': []}
    test  = {'data': [], 'label': []}
    val   = {'data': [], 'label': []}

    for file, label in zip(files, labels):
        data, label_ = time_frequency_transform(file, label, type, trans, img_size, normal, gray)

        if file.endswith("_0.mat") or file.endswith("_1.mat"):
            train['data'].append(data)
            train['label'].extend(label_)

        elif file.endswith("_2.mat"):
            test['data'].append(data)
            test['label'].extend(label_)

        elif file.endswith("_3.mat"):
            val['data'].append(data)
            val['label'].extend(label_)

    train['data'] = np.concatenate(train['data'], axis=0)
    test['data']  = np.concatenate(test['data'], axis=0)
    val['data']   = np.concatenate(val['data'], axis=0)

    train['label'] = np.asarray(train['label'])
    test['label']  = np.asarray(test['label'])
    val['label']   = np.asarray(val['label'])

    return train, test, val

def train_test_val_by_mixed_811(all_datasets):
    def train_test_split_load(dataset, test_size=0.2):  
        X_train, X_temp, y_train, y_temp = train_test_split(dataset['data'], dataset['label'], test_size=test_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        return {
            "train": {"data": X_train, "label": y_train},
            "val": {"data": X_val, "label": y_val},
            "test": {"data": X_test, "label": y_test},
        }
    
    splits = [train_test_split_load(ds) for ds in all_datasets]

    final = {"train": {"data": [], "label": []},
         "val": {"data": [], "label": []},
         "test": {"data": [], "label": []}}

    for split in splits:
        for key in ["train", "val", "test"]:
            final[key]["data"].append(split[key]["data"])
            final[key]["label"].append(split[key]["label"])

    for key in ["train", "val", "test"]:
        final[key]["data"] = np.concatenate(final[key]["data"], axis=0)
        final[key]["label"] = np.concatenate(final[key]["label"], axis=0)

    return final["train"], final["val"], final["test"]

def mixed_to_dinstinct_class(dataset):
    normal = {"data": [], "label": []}
    outer_race = {"data": [], "label": []}
    inner_race = {"data": [], "label": []}
    ball = {"data": [], "label": []}

    for data, label in zip(dataset['data'], dataset['label']):
        if label == 0:
            normal['data'].append(data)
            normal['label'].append(label)
        elif label == 1:
            outer_race['data'].append(data)
            outer_race['label'].append(label)
        elif label == 2:
            inner_race['data'].append(data)
            inner_race['label'].append(label)
        elif label == 3:
            ball['data'].append(data)
            ball['label'].append(label)

    for cls in [normal, outer_race, inner_race, ball]:
            cls["data"] = np.array(cls["data"])
            cls["label"] = np.array(cls["label"])

    return normal, outer_race, inner_race, ball

def save_data(data, save_path):
    tensor = torch.from_numpy(data['data'])
    labels = torch.from_numpy(data['label'])
    dict = {
        'data': tensor,
        'label': labels
    }
    torch.save(dict, save_path)