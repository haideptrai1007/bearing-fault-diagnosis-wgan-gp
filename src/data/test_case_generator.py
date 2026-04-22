import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.signal import resample_poly
import random
from collections import Counter


def get_mat(root_path, normal=False):
    """
    Read matlab file to get signal and perform resample for normal signal
    """
    file_list =  [os.path.join(root_path, mat) for mat in os.listdir(root_path)]

    signal_list = []
    for file in file_list:
        for i, j in loadmat(file).items():
            if "DE" in i:
                if normal:
                    signal_list.append(resample_poly(j.squeeze(), up=1, down=4))
                else:
                    signal_list.append(j.squeeze())
    return signal_list


def split_in_half(signal):
    """Split signal in half"""
    idx = len(signal) // 2
    return signal[:idx], signal[idx:]


def create_label_segment(signal, label):
    """Create corresponding label for each matched point"""
    return np.full(len(signal), label)


def normal_fault1_fault2(normal, normal_id, normal_label, fault1, fault2, fault_id_1, fault_id_2, fault_label1, fault_label2, root_path):
    """
    Concatenate normal signal with one or two fault types
    """
    tc_normal_h1, tc_normal_h2 = split_in_half(normal[normal_id])
    tc_fault1_h1, tc_fault1_h2 = split_in_half(fault1[fault_id_1])
    tc_fault2_h1, tc_fault2_h2 = split_in_half(fault2[fault_id_2])

    test_case_1 = np.concatenate(
        [tc_normal_h1, tc_fault1_h1, tc_normal_h2, tc_fault2_h1]
    )

    gt_1 = np.concatenate([
        create_label_segment(tc_normal_h1, normal_label),
        create_label_segment(tc_fault1_h1, fault_label1),
        create_label_segment(tc_normal_h2, normal_label),
        create_label_segment(tc_fault2_h1, fault_label2),
    ])

    test_case_2 = np.concatenate(
        [tc_normal_h2, tc_fault1_h2, tc_normal_h1, tc_fault2_h2]
    )

    gt_2 = np.concatenate([
        create_label_segment(tc_normal_h2, normal_label),
        create_label_segment(tc_fault1_h2, fault_label1),
        create_label_segment(tc_normal_h1, normal_label),
        create_label_segment(tc_fault2_h2, fault_label2),
    ])

    savemat(os.path.join(root_path, "test_case1.mat"),
            {"testcase1_DE_time": test_case_1,
             "groud_truth_DE_time": gt_1})

    savemat(os.path.join(root_path, "test_case2.mat"),
            {"testcase2_DE_time": test_case_2,
             "groud_truth_DE_time": gt_2})
    
def normal_all_fault(normal, normal_id, normal_label, fault1, fault2, fault3, fault_id_1, fault_id_2, fault_id_3, fault_label1, fault_label2, fault_label3, root_path):
    """
    Concatenate normal signal, outer race fault, inner race fault, ball fault into one signal
    """
    tc_normal_h1, tc_normal_h2 = split_in_half(normal[normal_id])
    tc_fault1_h1, tc_fault1_h2 = split_in_half(fault1[fault_id_1])
    tc_fault2_h1, tc_fault2_h2 = split_in_half(fault2[fault_id_2])
    tc_fault3_h1, tc_fault3_h2 = split_in_half(fault3[fault_id_3])

    test_case_1 = np.concatenate(
        [tc_normal_h1, tc_fault1_h1, tc_fault2_h1, tc_fault3_h1]
    )

    gt_1 = np.concatenate([
        create_label_segment(tc_normal_h1, normal_label),
        create_label_segment(tc_fault1_h1, fault_label1),
        create_label_segment(tc_fault2_h1, fault_label2),
        create_label_segment(tc_fault3_h1, fault_label3),
    ])

    test_case_2 = np.concatenate(
        [tc_normal_h2, tc_fault1_h2, tc_fault2_h2, tc_fault3_h2]
    )

    gt_2 = np.concatenate([
        create_label_segment(tc_normal_h2, normal_label),
        create_label_segment(tc_fault1_h2, fault_label1),
        create_label_segment(tc_fault2_h2, fault_label2),
        create_label_segment(tc_fault3_h2, fault_label3),
    ])

    savemat(os.path.join(root_path, "test_case1.mat"),
            {"testcase1_DE_time": test_case_1,
             "groud_truth_DE_time": gt_1})

    savemat(os.path.join(root_path, "test_case2.mat"),
            {"testcase2_DE_time": test_case_2,
             "groud_truth_DE_time": gt_2})