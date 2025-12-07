import os
import time
import pandas as pd
import numpy as np
import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, img_paths, img_labels, transform=None, get_aux=False, aux=None):
        self.root = root
        self.transform = transform
        self.get_aux = get_aux
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.aux = aux

    def __getitem__(self, idx):
        # start_time = time.time()
        img_paths = self.img_paths[idx]
        label = self.img_labels[idx]

        if isinstance(img_paths, str):
            img_paths = [img_paths]

        imgs = []
        for img_path in img_paths:
            img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        # print(f"耗时： {(time.time() - start_time)}秒")

        if len(imgs) == 1:
            imgs = imgs[0]

        if self.get_aux:
            return imgs, label, self.aux[idx]
        else:
            return imgs, label

    def __len__(self):
        return len(self.img_paths)

    def __distribute__(self):
        distribute_ = np.array(self.img_labels)
        return np.sum(distribute_ == 0), np.sum(distribute_ == 1), np.sum(distribute_ == 2), np.sum(distribute_ == 3),\
            np.sum(distribute_ == 4), np.sum(distribute_ == 5), np.sum(distribute_ == 6)


def get_triple_meta_data(df):
    on_paths = list(df.onset_frame_path)
    apex_paths = list(df.apex_frame_path)
    optical_flow_paths = list(df.optical_flow_path)

    paths = [(on, apex, optical_flow) for (on, apex, optical_flow) in zip(on_paths, apex_paths, optical_flow_paths)]
    labels = list(df.label)
    return paths, labels


def get_triple_meta_data_magnet(df):
    on_paths = list(df.onset_frame_path)
    apex_paths = list(df.apex_frame_path)
    magnet_paths = list(df.magnet_path)

    paths = [(on, apex, mag) for (on, apex, mag) in zip(on_paths, apex_paths, magnet_paths)]
    labels = list(df.label)
    return paths, labels


def get_apex_data(df):
    paths = list(df.apex_frame_path)
    labels = list(df.label)

    return paths, labels


def get_optical_flow_data(df):
    paths = list(df.optical_flow_path)
    labels = list(df.label)

    return paths, labels


def get_on_apex_data(df):
    on_paths = list(df.onset_frame_path)
    apex_paths = list(df.apex_frame_path)

    paths = [(on, apex) for (on, apex) in zip(on_paths, apex_paths)]
    labels = list(df.label)
    return paths, labels


def get_apex_optical_flow_data(df):
    apex_paths = list(df.apex_frame_path)
    optical_flow_paths = list(df.optical_flow_path)

    paths = [(apex, optical_flow) for (apex, optical_flow) in zip(apex_paths, optical_flow_paths)]
    labels = list(df.label)
    return paths, labels

def get_image_difference_data(df):
    paths = list(df.image_difference_path)
    labels = list(df.label)
    return paths, labels


def get_flow_diff_data(df):
    optical_flow_paths = list(df.optical_flow_path)
    image_difference_paths = list(df.image_difference_path)
    paths = [(optical_flow, diff) for (optical_flow, diff) in zip(optical_flow_paths, image_difference_paths)]
    labels = list(df.label)
    return paths, labels

def get_apex_data(df):
    paths = list(df.apex_frame_path)
    labels = list(df.label)
    return paths, labels


def get_three_meta_data(df):
    on_paths = list(df.onset_frame_path)
    apex_paths = list(df.apex_frame_path)
    off_paths = list(df.offset_frame_path)
    paths = [(on, apex, off) for (on, apex, off) in zip(on_paths, apex_paths, off_paths)]
    labels = list(df.label)
    return paths, labels


def get_four_meta_data(df):
    on_paths = list(df.onset_frame_path)
    apex_paths = list(df.apex_frame_path)
    off_paths = list(df.offset_frame_path)
    optical_flow_paths = list(df.optical_flow_path)
    paths = [(on, apex, off, optical_flow) for (on, apex, off, optical_flow) in
             zip(on_paths, apex_paths, off_paths, optical_flow_paths)]
    labels = list(df.label)
    return paths, labels


def data_split(file_path, subject_out_idx=0):
    """Split dataset into train set and validation set
	"""

    data_sub_column = 'data_sub'

    df = pd.read_csv(file_path)
    subject_list = list(df[data_sub_column].unique())
    subject_out = subject_list[subject_out_idx]
    print('subject_out', subject_out)
    df_train = df[df[data_sub_column] != subject_out]
    df_val = df[df[data_sub_column] == subject_out]

    return df_train, df_val


def data_split_v2(file_path):
    df = pd.read_csv(file_path)

    # 计算前90%数据的行数
    total_rows = len(df)
    print("total_rows: ", total_rows)
    rows_90_percent = int(0.9 * total_rows)

    # 取出前90%的数据
    df_train = df.iloc[:rows_90_percent]
    df_val = df.iloc[rows_90_percent:]

    print("训练集大小:", df_train.shape[0])
    print("验证集大小:", df_val.shape[0])

    return df_train, df_val


# def upsample_subdata(df, df_four, number=4):
#     result = df.copy()
#     for i in range(df.shape[0]):
#         quotient = int(number)  # 确保是整数
#         remainder = number % 1
#         remainder = 1 if np.random.rand() < remainder else 0
#         value = quotient + remainder

#         tmp = df_four[(df_four['subject'] == df.iloc[i]['subject']) & (df_four['filename'] == df.iloc[i]['filename'])]
#         value = min(value, tmp.shape[0])
#         tmp = tmp.sample(int(value))
#         result = pd.concat([result, tmp])
#     return result


# def sample_data(df, df_four, num_classes, scale_factor=1):
#     # 分离7个类别的数据
#     df_list = [df[df.label == i] for i in range(num_classes)]

#     # 选择一个基准类别（通常选择数量最多的类别）
#     base_class_size = max([sub_df.shape[0] for sub_df in df_list]) * scale_factor

#     # 计算每个类别的上采样比例
#     upsample_ratios = [(base_class_size / sub_df.shape[0]) - 1 if sub_df.shape[0] < base_class_size else 0 for sub_df in
#                        df_list]

#     # 上采样每个类别的数据
#     for i in range(num_classes):
#         df_list[i] = upsample_subdata(df_list[i], df_four, upsample_ratios[i])
#         print(f'df_class_{i}', df_list[i].shape)

#     # 合并所有上采样后的类别数据
#     df_balanced = pd.concat(df_list)
#     return df_balanced











# def upsample_subdata(df, df_four, target_size):
#     result = df.copy()
#     current_size = df.shape[0]
    
#     for i in range(current_size):
#         if result.shape[0] >= target_size:
#             break
        
#         quotient = int(target_size / current_size)  # 确保是整数
#         remainder = (target_size / current_size) % 1
#         remainder = 1 if np.random.rand() < remainder else 0
#         value = quotient + remainder

#         tmp = df_four[(df_four['subject'] == df.iloc[i]['subject']) & (df_four['filename'] == df.iloc[i]['filename'])]
#         value = min(value, tmp.shape[0])
#         tmp = tmp.sample(int(value))
#         result = pd.concat([result, tmp])
    
#     # Ensure we don't exceed the target size
#     if result.shape[0] > target_size:
#         result = result.sample(target_size)
    
#     return result

# def sample_data(df, df_four, num_classes):
#     # 分离各个类别的数据
#     df_list = [df[df.label == i] for i in range(num_classes)]
    
#     # 计算各个类别的总数量（df + df_four）
#     total_sizes = [df[df.label == i].shape[0] + df_four[df_four.label == i].shape[0] for i in range(num_classes)]
    
#     # 找到最小的总数量
#     min_size = min(total_sizes)
#     print(min_size, total_sizes)
    
#     # 调整每个类别的数据到最小的总数量
#     for i in range(num_classes):
#         combined_size = df_list[i].shape[0] + df_four[df_four.label == i].shape[0]
#         target_size = min_size - df_list[i].shape[0]
#         if target_size > 0 and combined_size >= min_size:
#             df_list[i] = upsample_subdata(df_list[i], df_four, target_size + df_list[i].shape[0])
#         elif combined_size < min_size:
#             # 如果 df + df_four 的总数小于 min_size，直接使用所有数据
#             df_list[i] = pd.concat([df_list[i], df_four[df_four.label == i]])
#         print(f'df_class_{i}', df_list[i].shape)
    
#     # 合并所有上采样后的类别数据
#     df_balanced = pd.concat(df_list)
#     return df_balanced



















# def upsample_subdata(df, df_four, number=4):
#     result = df.copy()
#     for i in range(df.shape[0]):
#         quotient = number // 1
#         remainder = number % 1
#         remainder = 1 if np.random.rand() < remainder else 0
#         value = quotient + remainder

#         tmp = df_four[df_four['data_sub'] == df.iloc[i]['data_sub']]
#         tmp = tmp[tmp['filename'] == df.iloc[i]['filename']]
#         value = min(value, tmp.shape[0])
#         tmp = tmp.sample(int(value))
#         result = pd.concat([result, tmp])
#     return result


# def sample_data(df, df_four):
#     df_neg = df[df.label == 0]
#     df_pos = df[df.label == 1]
#     df_sur = df[df.label == 2]

#     num_sur = 4
#     num_pos = 5 * df_sur.shape[0] / df_pos.shape[0] - 1
#     num_neg = 5 * df_sur.shape[0] / df_neg.shape[0] - 1
#     print(num_sur, num_pos, num_neg)

#     df_neg = upsample_subdata(df_neg, df_four, num_neg)
#     df_pos = upsample_subdata(df_pos, df_four, num_pos)
#     df_sur = upsample_subdata(df_sur, df_four, num_sur)
#     print('df_neg', df_neg.shape)
#     print('df_pos', df_pos.shape)
#     print('df_sur', df_sur.shape)

#     df = pd.concat([df_neg, df_pos, df_sur])
#     return df





def upsample_subdata(df, df_four, number=4):
    result = df.copy()
    for i in range(df.shape[0]):
        quotient = int(number)
        remainder = number % 1
        remainder = 1 if np.random.rand() < remainder else 0
        value = quotient + remainder

        tmp = df_four[df_four['data_sub'] == df.iloc[i]['data_sub']]
        tmp = tmp[tmp['filename'] == df.iloc[i]['filename']]
        value = min(value, tmp.shape[0])
        tmp = tmp.sample(int(value))
        result = pd.concat([result, tmp])
    return result


def sample_data(df, df_four):
    # 分割数据集为不同类别
    categories = df['label'].unique()
    
    # 初始化字典以存储每个类别的数据
    category_data = {cat: df[df.label == cat] for cat in categories}
    
    # 以每个类别的数据量作为基准进行平衡，设定一个合理的目标比例
    base_class = min(category_data, key=lambda x: len(category_data[x]))
    base_data = category_data[base_class]
    num_base_samples = 4  # 假设我们希望每个基准类别的数据行增加 4 个样本
    
    # 计算每个类别需要增加的样本数量，确保结果为正
    num_samples = {}
    for cat in categories:
        if cat != base_class:
            # 计算目标比例为基准类别数据量的 5 倍
            target_ratio = (num_base_samples + 1) * base_data.shape[0] / category_data[cat].shape[0]
            num_samples[cat] = max(target_ratio - 1, 0)
        else:
            num_samples[cat] = num_base_samples
    
    print(num_samples)
    
    # 对每个类别进行上采样
    for cat in categories:
        if num_samples[cat] > 0:
            category_data[cat] = upsample_subdata(category_data[cat], df_four, num_samples[cat])
        print(f'df_{cat}', category_data[cat].shape)
    
    # 合并所有类别的数据
    df = pd.concat(category_data.values())
    return df
