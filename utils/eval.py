import numpy as np
import torch

def compute_errors(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    return mse.item(), rmse.item(), mae.item()


def valid(config, model, val_generator, device, dataset=None):
    model.eval()

    city = config['data']['city']

    mean_mse_loss_pickup = []
    mean_mae_loss_pickup = []
    mean_rmse_loss_pickup = []

    mean_mse_loss_dropoff = []
    mean_mae_loss_dropoff = []
    mean_rmse_loss_dropoff = []

    for i, (X, Y, meta) in enumerate(val_generator):
        X_c = X[0].type(torch.FloatTensor).to(device)
        X_p = X[1].type(torch.FloatTensor).to(device)
        X_t = X[2].type(torch.FloatTensor).to(device)
        TS_c = meta[0].type(torch.FloatTensor).to(device)
        TS_p = meta[1].type(torch.FloatTensor).to(device)
        TS_t = meta[2].type(torch.FloatTensor).to(device)
        TS_Y = meta[3][:, :8].type(torch.FloatTensor).to(device)
        pois = meta[4].type(torch.FloatTensor).to(device)
        Y = Y.type(torch.FloatTensor).to(device)

        # Forward pass
        X = torch.cat((X_c, X_p, X_t), dim=1)
        TS = torch.cat((TS_c, TS_p, TS_t), dim=1)
        pred, _, _, _, _ = model(X, TS, pois)

        pred = pred.cpu().data.numpy()
        Y = Y.cpu().data.numpy()


        pred_pickup = dataset.denormalize_train(pred[:, 0], 0)
        Y_pickup = dataset.denormalize_train(Y[:, 0], 0)
        pred_dropoff = dataset.denormalize_train(pred[:, 1], 1)
        Y_dropoff = dataset.denormalize_train(Y[:, 1], 1)

        mse, rmse, mae = compute_errors(Y_pickup, pred_pickup)

        mean_mse_loss_pickup.append(mse)
        mean_rmse_loss_pickup.append(rmse)
        mean_mae_loss_pickup.append(mae)

        mse, rmse, mae = compute_errors(Y_dropoff, pred_dropoff)

        mean_mse_loss_dropoff.append(mse)
        mean_rmse_loss_dropoff.append(rmse)
        mean_mae_loss_dropoff.append(mae)


    mean_mse_loss_pickup = np.mean(mean_mse_loss_pickup)
    mean_rmse_loss_pickup = np.mean(mean_rmse_loss_pickup)
    mean_mae_loss_pickup = np.mean(mean_mae_loss_pickup)

    mean_mse_loss_dropoff = np.mean(mean_mse_loss_dropoff)
    mean_rmse_loss_dropoff = np.mean(mean_rmse_loss_dropoff)
    mean_mae_loss_dropoff = np.mean(mean_mae_loss_dropoff)

    return mean_mse_loss_pickup, mean_rmse_loss_pickup, mean_mae_loss_pickup, \
           mean_mse_loss_dropoff, mean_rmse_loss_dropoff, mean_mae_loss_dropoff



