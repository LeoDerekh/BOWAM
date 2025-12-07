import os
import time
from os.path import join

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from models.model_utils import get_model
from options import Options
from utils import calculate_metrics, train_one_epoch, evaluate, save_info_append, load_me_data, recognition_evaluation, get_optimizer, \
    get_scheduler


def train(opt, subject_out_idx):
    train_loader, val_loader = load_me_data(opt, subject_out_idx=subject_out_idx)

    model = get_model(opt)

    criterionCE = torch.nn.CrossEntropyLoss().to(opt.device)
    torch.nn.DataParallel(criterionCE, opt.gpu_ids)

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = get_optimizer(opt, pg)
    if opt.model == "mymodel2":
        optimizer = torch.optim.AdamW([
                        {'params': model.module.dgm.parameters(),'lr': 0.002},
                        {'params': model.module.main_branch.parameters(), 'lr': opt.lr}], betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    else:
        optimizer = get_optimizer(opt, pg)
    scheduler = get_scheduler(opt, optimizer)

    train_loss_bank = []
    best_cor = -1
    best_total = -1
    best_acc = -1
    best_uf1 = -1
    best_epoch = 0
    best_targets_total = []
    best_pred_targets_total = []
    strat_time = time.strftime("%Y-%m-%d %H:%M:%S")

        
    for epoch in range(1, opt.epochs + 1):
        # train
        train_loss = train_one_epoch(opt=opt,
                                     model=model,
                                     criterion=criterionCE,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=opt.device,
                                     epoch=epoch)
        train_loss_bank.append(train_loss)

        scheduler.step()


        # validate
        cor, total, acc, uf1, targets_total, pred_targets_total, val_loss = evaluate(opt=opt,
                                                                                     model=model,
                                                                                     criterion=criterionCE,
                                                                                     data_loader=val_loader,
                                                                                     device=opt.device,
                                                                                     epoch=epoch)

        # save checkpoint if needed
        # is_best = uf1 >= best_uf1
        # best_uf1 = max(uf1, best_uf1)
        # if is_best:
        #     is_best = acc > best_acc
        #     best_acc = max(acc, best_acc)
        # if is_best:
        #     best_epoch = epoch
        #     best_targets_total = targets_total
        #     best_pred_targets_total = pred_targets_total
        #     best_cor = cor
        #     best_total = total
        #     # save_ckpt()
        #     torch.save(model.state_dict(), join(opt.ckpt_dir, "best_model_{}.pth".format(subject_out_idx)))
        #     print('!' * 100, "model saved")


        # method2
        if uf1 > best_uf1:
            best_uf1 = uf1
            best_acc = acc
            best_epoch = epoch
            best_targets_total = targets_total
            best_pred_targets_total = pred_targets_total
            best_cor = cor
            best_total = total
            # save_ckpt()
            torch.save(model.state_dict(), join(opt.ckpt_dir, "best_model_{}.pth".format(subject_out_idx)))
            print('!' * 100, "model saved")

        if best_acc == 100:
            break
    print('best_targets_total     :', best_targets_total)
    print('best_pred_targets_total:', best_pred_targets_total)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    best_info = {'sub_val': opt.sub_val, 'best_epoch': best_epoch,
                 'best_cor': best_cor, 'best_total': best_total,
                 'best_acc': best_acc, 'best_uf1': best_uf1,
                 'best_targets_total': str(best_targets_total),
                 'best_pred_targets_total': str(best_pred_targets_total),
                 'strat_time': strat_time, 'end_time': end_time,
                 'path': join(opt.ckpt_dir,
                              opt.model + '_' + opt.dataset + '_sub' + str(opt.sub_val).zfill(2) + '.pth')}
    best_info_path = os.path.join(opt.ckpt_dir, "best_info.csv")
    save_info_append(best_info_path, best_info)
    print('best_epoch:', best_epoch, 'best_acc:', best_acc, 'best_uf1:', best_uf1)

    return best_targets_total, best_pred_targets_total


def train_loso(opt):
    strat_time = time.strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(opt.data_apex_frame_path)
    subject_list = list(df['data_sub'].unique())
    subject_num = len(subject_list)
    pred_targets_total_loso = []
    targets_total_loso = []
    for i in range(subject_num):
        opt.sub_val = i
        best_targets_total, best_pred_targets_total = train(opt, i)
        print('Best Predicted    :', best_pred_targets_total)
        print('Best Ground Truth :', best_targets_total)
        print('Evaluation until this subject: ')
        targets_total_loso += best_targets_total
        pred_targets_total_loso += best_pred_targets_total

        uf1 = f1_score(targets_total_loso, pred_targets_total_loso, average='macro')
        uar = recall_score(targets_total_loso, pred_targets_total_loso, average='macro')
        print('UF1: {:.4f}, UAR: {:.4f}'.format(uf1, uar))
        UF1, UAR, ACC, class_accuracies = calculate_metrics(targets_total_loso, pred_targets_total_loso, opt.num_classes)
        print('UF1: {:.4f}, UAR: {:.4f}, ACC: {:.4f}, class_accuracies: {}'.format(UF1, UAR, ACC, class_accuracies))

    end_time = time.strftime("%Y-%m-%d %H:%M:%S")

    print('targets_total_loso:      ', targets_total_loso)
    print('pred_targets_total_loso: ', pred_targets_total_loso)
    y_true = targets_total_loso
    y_pred = pred_targets_total_loso
    cor = accuracy_score(y_true, y_pred, normalize=False)
    total = len(y_true)
    acc = accuracy_score(y_true, y_pred) * 100
    uf1 = f1_score(y_true, y_pred, average='macro')
    uar = recall_score(y_true, y_pred, average='macro')
    model = opt.model
    dataset = opt.dataset
    num_classes = opt.num_classes
    pretrained = opt.pretrained
    epochs = opt.epochs
    batch_size = opt.batch_size
    lr = opt.lr
    optimizer = opt.optimizer
    lr_policy = opt.lr_policy
    norm = opt.norm
    init_type = opt.init_type
    init_gain = opt.init_gain
    ckpt_dir = opt.ckpt_dir
    gpu_ids = str(opt.gpu_ids)
    lucky_seed = opt.lucky_seed
    data_apex_frame_path = opt.data_apex_frame_path
    data_n_frames_path = opt.data_n_frames_path

    info_dict = {'model': model, 'dataset': dataset, 'num_classes': num_classes,
                 'pretrained': pretrained, 'cor': cor, 'total': total, 'Acc': acc, 'UF1': uf1, 'UAR': uar,
                 'epochs': epochs, 'batch_size': batch_size, 'lr': lr, 'optimizer': optimizer,
                 'lr_policy': lr_policy,
                 'norm': norm, 'init_type': init_type, 'init_gain': init_gain,
                 'ckpt_dir': ckpt_dir, 'gpu_ids': gpu_ids, 'lucky_seed': lucky_seed,
                 'data_apex_frame_path': data_apex_frame_path,
                 'data_n_frames_path': data_n_frames_path,
                 'strat_time': strat_time,
                 'end_time': end_time}
    info_path = join(opt.ckpt_dir, "result_loso.csv")
    save_info_append(info_path, info_dict)
    result_path = join(opt.results, "result_5.csv")
    save_info_append(result_path, info_dict)

    print('LOSO: Accuracy: {:d}/{} ({:.2f}%), UF1: {:.4f}, UAR: {:.4f}'.format(cor, total, acc, uf1, uar))

    print('Final Evaluation: ')
    UF1, UAR, ACC, class_accuracies = calculate_metrics(y_true, y_pred, opt.num_classes)
    print('UF1: {:.4f}, UAR: {:.4f}, ACC: {:.4f}, class_accuracies: {}'.format(UF1, UAR, ACC, class_accuracies))


if __name__ == '__main__':
    opt = Options().parse()
    start_time = time.time()
    if opt.mode == "test":
        best_targets_total, best_pred_targets_total = train(opt, opt.sub_val)
    else:
        train_loso(opt)
    print(f"耗时： {(time.time() - start_time) / 3600 :.2f}小时")
    print('[THE END]')
