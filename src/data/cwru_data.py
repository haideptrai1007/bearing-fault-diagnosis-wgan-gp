import os
import scipy.io
import numpy as np
from scipy.signal import resample_poly
from src.utils.utils import sliding_window

def cwru_load_data(root_path, normal=False):
    files, labels = [], []
    if normal:
        for file in os.listdir(root_path):
            full_path = os.path.join(root_path, file)
            if file.endswith(".mat") and os.path.isfile(full_path):
                files.append(full_path)
                labels.append("Normal")
    else:
        for fault in os.listdir(root_path):
            fault_path = os.path.join(root_path, fault)
            if not os.path.isdir(fault_path):
                continue
            for diameter in os.listdir(fault_path):
                diameter_path = os.path.join(fault_path, diameter)
                if not os.path.isdir(diameter_path):
                    continue
                for root, _, mat_files in os.walk(diameter_path):
                    for mat_file in mat_files:
                        if mat_file.endswith(".mat"):
                            files.append(os.path.join(root, mat_file))
                            labels.append(fault)
    return files, labels


def cwru_transform(file, label, transform, type="DE", window_size=2048, overlap=0.5, img_size=128, normal=False, gray=False):
    mat_file = scipy.io.loadmat(file)
    images = []
    labels = []
    for i, j in mat_file.items():
        if type in i:
            signal = j.squeeze()
            if normal:
                signal = resample_poly(signal, up=1, down=4)

            windows = sliding_window(signal, window_size, overlap)
            for window in windows:
                img = transform(window, img_size=img_size, gray=gray)
                images.append(img)
                labels.append(label)
    
    label_map = {"Normal": 0, "OR": 1, "IR": 2, "B": 3}
    labels = np.asarray([label_map[label] for label in labels])

    return np.array(images), labels


def cwru_split(load_data, path, type, trans, window_size=2048, overlap=0.5, img_size=128, normal=False, gray=True):
    files, labels = load_data(path, normal)
    train = {'data': [], 'label': []}
    val   = {'data': [], 'label': []}
    test   = {'data': [], 'label': []}

    for file, label in zip(files, labels):
        data, label_ = cwru_transform(file, label, type, trans, window_size, overlap, img_size, normal, gray)

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
    test['data'] = np.concatenate(test['data'], axis=0)
    val['data']   = np.concatenate(val['data'], axis=0)

    train['label'] = np.asarray(train['label'])
    test['label'] = np.asarray(test['label'])
    val['label']   = np.asarray(val['label'])

    return train, test, val


def cwru_seperate(dataset):
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


def cwru_inference(file, transform, type="DE", window_size=2048, overlap=0, img_size=128, normal=False, gray=True):
    mat_file = scipy.io.loadmat(file)
    labels = {"OR", "IR", "B", "Normal"}
    images, grouth_truths = [], []
    for i, j in mat_file.items():
        if type in i:
            signal = j.squeeze()
            if normal:
                signal = resample_poly(signal, up=1, down=4)

            windows = sliding_window(signal, window_size, overlap)
            for window in windows:
                img = transform(window, img_size=img_size, gray=gray)
                images.append(img)
                grouth_truths.extend([part for part in file.split("\\") if part in labels])

    return np.array(images), np.array(grouth_truths)