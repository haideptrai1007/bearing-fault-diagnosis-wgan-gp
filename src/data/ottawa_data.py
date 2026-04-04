import scipy.io
from src.utils.utils import sliding_window
import numpy as np
import os


def ottawa_load_data(root_path):
    files, labels = [], []
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            files.append(file_path)
            if "Healthy" in folder:
                labels.append("N")
            elif "Inner" in folder:
                labels.append("IR")
            elif "Outer" in folder:
                labels.append("OR")
            elif "Ball" in folder:
                labels.append("B")
            elif "Cage" in folder:
                labels.append("C")
    return files, labels


def ottawa_transform(file, label, transform, window_size=2048, overlap=0.5, img_size=128, gray=False):
    mat_file = scipy.io.loadmat(file)
    images, labels = [], []

    tag = file.split("\\")[-1].split(".")[0]
    signal = mat_file[tag][:42000, 0].squeeze()

    windows = sliding_window(signal, window_size, overlap)

    for window in windows:
        img = transform(window, img_size=img_size, gray=gray)
        images.append(img)
        labels.append(label)

    label_map = {"N":0, "OR":1, "IR":2, "B":3, "C":4}
    labels = np.asarray([label_map[l] for l in labels])

    return np.array(images), labels

def ottawa_split(load_data, path, trans, window_size=2048, overlap=0.5, img_size=128, gray=False):
    files, labels = load_data(path)
    train = {'data': [], 'label': []}
    test = {'data': [], 'label': []}
    val = {'data': [], 'label': []}

    test_list = ['4', '9', '14', '19']
    val_list = ['5', '10', '15', '20']

    for file, label in zip(files, labels):
        tag = file.split('\\')[-1].split("_")[1]
        data, label_ = ottawa_transform(file, label, trans, window_size, overlap, img_size, gray)

        if tag in test_list:
            test['data'].append(data)
            test['label'].extend(label_)

        elif tag in val_list:
            val['data'].append(data)
            val['label'].extend(label_)

        else:
            train['data'].append(data)
            train['label'].extend(label_)
        

    train['data'] = np.concatenate(train['data'], axis=0)
    test['data'] = np.concatenate(test['data'], axis=0)
    val['data']   = np.concatenate(val['data'], axis=0)

    train['label'] = np.asarray(train['label'])
    test['label'] = np.asarray(test['label'])
    val['label']   = np.asarray(val['label'])

    return train, test, val

def ottawa_seperate(dataset):
    normal = {"data": [], "label": []}
    outer_race = {"data": [], "label": []}
    inner_race = {"data": [], "label": []}
    ball = {"data": [], "label": []}
    cage = {"data": [], "label": []}

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
        elif label == 4:
            cage['data'].append(data)
            cage['label'].append(label)

    for cls in [normal, outer_race, inner_race, ball, cage]:
            cls["data"] = np.array(cls["data"])
            cls["label"] = np.array(cls["label"])

    return normal, outer_race, inner_race, ball, cage