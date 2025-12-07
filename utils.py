import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from torch.optim import lr_scheduler
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from torchvision import transforms
from sklearn.utils import shuffle
current_working_directory = os.getcwd()
print("Current working directory:", current_working_directory)
sys.path.append(current_working_directory)
from dataset import data_split, get_apex_data, get_flow_diff_data, get_four_meta_data, get_image_difference_data, get_on_apex_data, get_three_meta_data, sample_data, Dataset, get_optical_flow_data


def load_me_data(opt, subject_out_idx):
    df_train, df_val = data_split(opt.data_apex_frame_path, subject_out_idx)
    # train oversampling
    df_n_frames = pd.read_csv(opt.data_n_frames_path)
    df_train = sample_data(df_train, df_n_frames)
    # df_train = sample_data(df_train, df_n_frames, num_classes=opt.num_classes, scale_factor=opt.scale_factor)
    # df_train = sample_data(df_train, df_n_frames, num_classes=opt.num_classes)
    df_train = shuffle(df_train)

    train_paths, train_labels = get_optical_flow_data(df_train)
    val_paths, val_labels = get_optical_flow_data(df_val)


    train_transforms = transforms.Compose([transforms.Resize((240, 240), interpolation=InterpolationMode.BICUBIC),
                                        transforms.RandomRotation(degrees=(-8, 8)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                saturation=0.2, hue=0.2),
                                        transforms.RandomCrop((224, 224)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                        ])

    val_transforms = transforms.Compose([transforms.Resize((240, 240), interpolation=InterpolationMode.BICUBIC),
                                        transforms.RandomRotation(degrees=(-8, 8)),
                                        # transforms.RandomHorizontalFlip(),
                                        # transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                # saturation=0.2, hue=0.2),
                                        transforms.CenterCrop((224, 224)),
                                        # transforms.RandomCrop((224, 224)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                        ])

    train_dataset = Dataset(root=opt.data_root,
                            img_paths=train_paths,
                            img_labels=train_labels,
                            transform=train_transforms)
    print('Train set size:', train_dataset.__len__())
    print('The Train dataset distribute:', train_dataset.__distribute__())

    val_dataset = Dataset(root=opt.data_root,
                          img_paths=val_paths,
                          img_labels=val_labels,
                          transform=val_transforms)
    print('Validation set size:', val_dataset.__len__())
    print('The Validation dataset distribute:', val_dataset.__distribute__())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=opt.batch_size,
                                               num_workers=opt.n_workers,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=opt.batch_size,
                                             num_workers=opt.n_workers,
                                             shuffle=False,
                                             pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(opt, model, criterion, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0
    train_bar = tqdm(data_loader, file=sys.stdout)

    for step, (inputs, labels) in enumerate(train_bar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        cur_lr = optimizer.param_groups[0]['lr']

        info_dict = {'sub_val': opt.sub_val, 'epoch': epoch, 'epochs': opt.epochs,
                     'epoch_steps': step * opt.batch_size,
                     'epoch_steps_len': len(data_loader.dataset),
                     'cur_lr': cur_lr,
                     'log_path': os.path.join(opt.ckpt_dir, opt.log_file),
                     'loss': loss}
        msg = '[Train: leave subject {} out][Epoch: {:0>3}/{:0>3}; Samples: {:0>4}/{:0>4}; LR: {:.7f}; Loss: {:.4f}] '.format(
            info_dict['sub_val'],
            info_dict['epoch'], info_dict['epochs'],
            info_dict['epoch_steps'], info_dict['epoch_steps_len'],
            info_dict['cur_lr'], info_dict['loss'])
        train_bar.desc = msg

    train_loss = running_loss / len(data_loader.dataset)
    return train_loss


test_total_steps = 0


def evaluate(opt, model, criterion, data_loader, device, epoch):
    model.eval()
    running_loss = 0
    targets_total = []
    pred_targets_total = []
    val_bar = tqdm(data_loader, file=sys.stdout)

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_bar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            y = labels.tolist()
            # print("y:", y)
            y_pred = preds.tolist()
            # print("y_pred:", y_pred)

            acc = accuracy_score(y, y_pred) * 100.0
            uf1 = f1_score(y, y_pred, average='macro')
            targets_total += y
            pred_targets_total += y_pred
            val_loss = running_loss / len(data_loader.dataset)
            global test_total_steps
            test_total_steps += opt.batch_size

            info_dict = {'sub_val': opt.sub_val, 'epoch': epoch, 'epochs': opt.epochs,
                        'epoch_steps': step * opt.batch_size,
                        'epoch_steps_len': len(data_loader.dataset),
                        'acc': acc, 'uf1': uf1,
                        'batch_size': opt.batch_size,
                        'log_path': os.path.join(opt.ckpt_dir, opt.log_file),
                        'loss': loss}
            msg = '[Test: leave subject {} out][Epoch: {:0>3}/{:0>3}; Samples: {:0>4}/{:0>4}; Acc: {:.2f}; UF1: {:.4f}; Loss: {:.4f}] '.format(
                info_dict['sub_val'],
                info_dict['epoch'], info_dict['epochs'],
                info_dict['epoch_steps'], info_dict['epoch_steps_len'],
                info_dict['acc'], info_dict['uf1'],
                info_dict['loss'])
            val_bar.desc = msg

    cor = accuracy_score(targets_total, pred_targets_total, normalize=False)
    total = len(targets_total)
    acc = accuracy_score(targets_total, pred_targets_total) * 100
    uf1 = f1_score(targets_total, pred_targets_total, average='macro')
    uar = recall_score(targets_total, pred_targets_total, average='macro')
    print('[Validation] Acc: {:.4f} | UF1: {:.4f} | UAR: {:.4f}'.format(acc, uf1, uar))

    # UF1, UAR, ACC, class_accuracies = calculate_metrics(targets_total, pred_targets_total, opt.num_classes)
    # print('[Evaluation] Acc: {:.4f} | UF1: {:.4f} | UAR: {:.4f} | class_accuracies: {}'.format(ACC, UF1, UAR, class_accuracies))
    # print(round(uf1, 8) == round(UF1, 8) and (round(uar, 8) == round(UAR, 8)))
    return cor, total, acc, uf1, targets_total, pred_targets_total, val_loss


def print_info_train(info_dict):
    msg = '[Train: leave subject {} out][Epoch: {:0>3}/{:0>3}; Samples: {:0>4}/{:0>4}; Time: {:.3f}s/Batch({}); LR: {:.7f}; Loss: {:.4f}] '.format(
        info_dict['sub_val'],
        info_dict['epoch'], info_dict['epochs'],
        info_dict['epoch_steps'], info_dict['epoch_steps_len'],
        info_dict['batch_size'],
        info_dict['cur_lr'], info_dict['loss'])
    print(msg)
    with open(info_dict['log_path'], 'a+') as f:
        f.write(msg + '\n')


def print_info_evaluate(info_dict):
    msg = '[Test: leave subject {} out][Epoch: {:0>3}/{:0>3}; Samples: {:0>4}/{:0>4}; Time: {:.3f}s/Batch({}); Acc: {:.2f}; UF1: {:.4f}; Loss: {:.4f}] '.format(
        info_dict['sub_val'],
        info_dict['epoch'], info_dict['epochs'],
        info_dict['epoch_steps'], info_dict['epoch_steps_len'],
        info_dict['batch_size'],
        info_dict['acc'], info_dict['uf1'],
        info_dict['loss'])
    print(msg)
    with open(info_dict['log_path'], 'a+') as f:
        f.write(msg + '\n')


def save_info_append(path, info):
    columns = [k for k in info.keys()]
    if not os.path.exists(path):
        df = pd.DataFrame(data=None, columns=columns)
        df.to_csv(path, index=False)
    df = pd.read_csv(path)
    new_row = pd.DataFrame(info, index=[0])
    df = pd.concat([df, new_row], ignore_index=True, axis=0)
    df.to_csv(path, index=False)


def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}

    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''


def get_optimizer(opt, parameters):
    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    elif opt.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError('optimizer [%s] is not implemented', opt.optimizer)
    return optimizer


def get_scheduler(opt, optimizer):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=opt.exponential_gamma)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def calculate_metrics(y_true, y_pred, num_classes):
    """
    计算UF1, UAR, ACC指标以及每个类别的准确率.

    参数:
    y_true (list or np.array): 真实标签.
    y_pred (list or np.array): 预测标签.
    num_classes (int): 类别总数.

    返回:
    tuple: 包含UF1, UAR, ACC和每个类别准确率的元组.
    """
    # 确保输入是np.array类型
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 检查num_classes是否大于零
    if num_classes <= 0:
        raise ValueError("num_classes必须大于零")

    # 初始化计数器
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)
    N = np.zeros(num_classes)

    # 计算TP, FP, FN, N
    for i in range(len(y_true)):
        true_class = y_true[i]
        pred_class = y_pred[i]
        N[true_class] += 1
        if true_class == pred_class:
            TP[true_class] += 1
        else:
            FP[pred_class] += 1
            FN[true_class] += 1

    # 计算每个类别的F1-score
    F1_scores = np.zeros(num_classes)
    valid_classes = N > 0  # 仅计算在 y_true 中出现的类别
    for i in range(num_classes):
        if valid_classes[i]:
            denom = (2 * TP[i] + FP[i] + FN[i])
            if denom > 0:
                F1_scores[i] = (2 * TP[i]) / denom

    # 计算UF1
    UF1 = np.mean(F1_scores[valid_classes])

    # 计算UAR
    recall_scores = np.zeros(num_classes)
    for i in range(num_classes):
        if N[i] > 0:
            recall_scores[i] = TP[i] / N[i]
    UAR = np.mean(recall_scores[valid_classes])

    # 计算ACC
    ACC = np.sum(TP) / np.sum(N)

    # 计算每个类别的准确率
    class_accuracies = np.zeros(num_classes)
    for i in range(num_classes):
        if N[i] > 0:
            class_accuracies[i] = TP[i] / N[i]

    return UF1, UAR, ACC, class_accuracies
