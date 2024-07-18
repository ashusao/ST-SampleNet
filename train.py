import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data_manager.helper import create_generators
from data_manager.st_dataset import STDataset

from model.STSampleNet import STSampleNet

from utils.early_stopping import EarlyStopping
from utils.eval import valid


def train(config):

    model_name = config['model']['name']
    # create checkpoint and run dirs if not exist
    out_dir = config['model']['dir'] + model_name + '/' + config['model']['exp']
    chkpt_dir = out_dir + '/checkpoint'
    writer_dir = out_dir + '/runs'
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(writer_dir, exist_ok=True)

    # set seed for reproducbility
    seed = int(config['general']['seed'])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # tensorboard writer
    writer = SummaryWriter(writer_dir)

    len_c, len_p, len_t = int(config['data']['closeness']), int(config['data']['period']), int(config['data']['trend'])
    map_h, map_w = int(config['grid']['height']), int(config['grid']['width'])
    train_teacher = config.getboolean('general', 'train_teacher')
    city = config['data']['city']
    n_c = int(config['data']['n_channel'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = STDataset(config=config, mode='train')
    train_generator, val_generator = create_generators(config=config, mode='train', dataset=train_dataset,
                                                       val_split=float(config['data']['val_split']))
    X, Y, meta = next(iter(train_generator))
    X_c = X[0].type(torch.FloatTensor).to(device)
    X_p = X[1].type(torch.FloatTensor).to(device)
    X_t = X[2].type(torch.FloatTensor).to(device)
    TS_c = meta[0].type(torch.FloatTensor).to(device)
    TS_p = meta[1].type(torch.FloatTensor).to(device)
    TS_t = meta[2].type(torch.FloatTensor).to(device)
    pois = meta[4].type(torch.FloatTensor).to(device)
    TS_Y_str = meta[5]

    print(TS_Y_str)
    #print(X_c.shape,  Y.shape, TS_c.shape, TS_Y.shape, pois.shape)

    # create model
    n_layer_spatial, n_layer_temporal, n_head_spatial, n_head_temporal, dim_feat = \
        int(config['stsamplenet']['n_layer_spatial']), \
        int(config['stsamplenet']['n_layer_temporal']), \
        int(config['stsamplenet']['n_head_spatial']), \
        int(config['stsamplenet']['n_head_temporal']), \
        int(config['stsamplenet']['embed_dim'])

    l1, l2, l3, l4 = float(config['stsamplenet']['prop_l1']), float(config['stsamplenet']['prop_l2']), \
                     float(config['stsamplenet']['prop_l3']), float(config['stsamplenet']['prop_l4'])
    region_keep_rate= float(config['stsamplenet']['region_keep_rate'])
    tau = float(config['stsamplenet']['tau'])

    teacher = STSampleNet(len_conf=(len_c, len_p, len_t), n_c=n_c, n_poi=10, embed_dim=dim_feat, map_w=map_w,
                          map_h=map_h,
                          dim_ts_feat=10, n_head_spatial=n_head_spatial, n_head_temporal=n_head_temporal,
                          n_layer_spatial=n_layer_spatial, n_layer_temporal=n_layer_temporal,
                          dropout=0.1, hirerachy_ratio=(l1, l2, l3, l4), region_keep_rate=region_keep_rate,
                          tau=tau, city=city, teacher=True, device=device, dir=config['data']['dir'])
    teacher.to(device)

    if train_teacher:
        model = teacher
    else:
        model = STSampleNet(len_conf=(len_c, len_p, len_t), n_c=n_c, n_poi=10, embed_dim=dim_feat, map_w=map_w, map_h=map_h,
                            dim_ts_feat=10, n_head_spatial=n_head_spatial, n_head_temporal=n_head_temporal,
                            n_layer_spatial=n_layer_spatial, n_layer_temporal=n_layer_temporal,
                            dropout=0.1, hirerachy_ratio=(l1, l2, l3, l4), region_keep_rate=region_keep_rate,
                            tau=tau, city=city, teacher=False, device=device, dir=config['data']['dir'])
        model.to(device)
    X = torch.cat((X_c, X_p, X_t), dim=1)
    TS = torch.cat((TS_c, TS_p, TS_t), dim=1)

    writer.add_graph(model, [X, TS, pois])

    n_epoch = int(config['train']['n_epoch'])
    lr = float(config['train']['lr'])
    patience = int(config['train']['patience'])
    alpha1 = float(config['train']['alpha1'])
    alpha2 = float(config['train']['alpha2'])
    epoch_save = [0, n_epoch - 1] + list(range(0, n_epoch, 50))  # 1*1000

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    es = EarlyStopping(patience=patience, mode='min', model=model, save_path=chkpt_dir + '/model.best.pth')

    if not train_teacher:
        teacher_location = config['model']['dir'] + model_name + '/' + config['model'][
            'teacher_exp'] + '/checkpoint/model.best.pth'
        print('Loading Teacher: ', teacher_location)
        teacher.load_state_dict(torch.load(teacher_location, map_location=lambda storage, loc: storage))

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
    mse_loss.to(device)
    l1_loss.to(device)

    print('==== Starting Training ====')

    for e in range(n_epoch):
        model.train()
        for i, (X, Y, meta) in enumerate(train_generator):

            X_c = X[0].type(torch.FloatTensor).to(device)
            X_p = X[1].type(torch.FloatTensor).to(device)
            X_t = X[2].type(torch.FloatTensor).to(device)
            TS_c = meta[0].type(torch.FloatTensor).to(device)
            TS_p = meta[1].type(torch.FloatTensor).to(device)
            TS_t = meta[2].type(torch.FloatTensor).to(device)
            pois = meta[4].type(torch.FloatTensor).to(device)

            Y = Y.type(torch.FloatTensor).to(device)

            X = torch.cat((X_c, X_p, X_t), dim=1)
            TS = torch.cat((TS_c, TS_p, TS_t), dim=1)
            outputs, cls_region_student, _, _, _ = model(X, TS, pois)
            outputs_teacher, cls_region_teacher, _, _, _ = teacher(X, TS, pois)

            if train_teacher:
                loss = mse_loss(outputs, Y) + alpha1 * l1_loss(outputs, Y)
            else:

                loss_kl = torch.mean(torch.stack([
                    kl_loss(
                        F.log_softmax(cls_region_student[:, i, :], dim=-1),
                        F.log_softmax(cls_region_teacher[:, i, :], dim=-1)
                    ) for i in range(cls_region_student.size(1))
                ]))

                #print(loss_kl)
                loss = mse_loss(outputs, Y) + alpha1 * l1_loss(outputs, Y) + alpha2 * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        v_mse_PU, v_rmse_PU, v_mae_PU, \
        v_mse_DO, v_rmse_DO, v_mae_DO = \
            valid(model, val_generator, device, train_dataset)

        print('Epoch [{}/{}], PU Val MSE: {:.4f} RMSE: {:.4f} MAE: {:.4f} '
              .format(e + 1, n_epoch, v_mse_PU, v_rmse_PU, v_mae_PU))
        print('Epoch [{}/{}], DO Val MSE: {:.4f} RMSE: {:.4f} MAE: {:.4f} '
              .format(e + 1, n_epoch, v_mse_DO, v_rmse_DO, v_mae_DO))

        writer.add_scalars('loss', {'val_vol': v_mse_PU,
                                    'val_inflow': v_mse_DO}, e)

        v_mse = np.mean([v_mse_PU, v_mse_DO])
        if es.step(v_mse, e):
            print('early stopped! With val loss:', v_mse)
            break

        if e in epoch_save:
            torch.save(model.state_dict(), chkpt_dir + '/%08d_model.pth' % (e))
            torch.save({
                    'optimizer': optimizer.state_dict(),
                    'epoch': e,
                }, chkpt_dir + '/%08d_optimizer.pth' % (e))

            print(chkpt_dir + '/%08d_model.pth' % (e) +' saved!')

    print('==== Training Finished ====')

    # create generators
    test_dataset = STDataset(config=config, mode='test')
    test_generator = create_generators(config=config, mode='test', dataset=test_dataset)

    best_model = chkpt_dir + '/model.best.pth'

    model.load_state_dict(torch.load(best_model, map_location=lambda storage, loc: storage))
    model.to(device)
    model.eval()

    train_mse_PU, train_rmse_PU, train_mae_PU, \
    train_mse_DO, train_rmse_DO, train_mae_DO = \
        valid(model, train_generator, device, train_dataset)

    test_mse_PU, test_rmse_PU, test_mae_PU, \
    test_mse_DO, test_rmse_DO, test_mae_DO = \
        valid(model, test_generator, device, test_dataset)

    print(
        'PU Train MSE: {:.4f} Train RMSE: {:.4f} Train MAE: {:.4f} Test MSE: {:.4f} Test RMSE: {:.4f} Test MAE: {:.4f}'
        .format(train_mse_PU, train_rmse_PU, train_mae_PU, test_mse_PU, test_rmse_PU, test_mae_PU))

    print(
        'DO Train MSE: {:.4f} Train RMSE: {:.4f} Train MAE: {:.4f} Test MSE: {:.4f} Test RMSE: {:.4f} Test MAE: {:.4f}'
        .format(train_mse_DO, train_rmse_DO, train_mae_DO, test_mse_DO, test_rmse_DO, test_mae_DO))

