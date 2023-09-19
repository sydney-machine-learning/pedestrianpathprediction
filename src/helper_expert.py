import torch
import numpy as np

from src.soft_dtw_cuda import *
from sklearn.cluster import KMeans

# from kmeans_pytorch import kmeans


def expert_find(test_batch_velocity, test_traj_abs, train_velocity, train_traj_abs, obs_length, step, gamma=2.0):
    """
    weighted: weather or not weight the input sequence for comparing DWT measurements;
    """
    global mse, criterion
    all_min_end = []
    rest_diff = []

    mse = torch.nn.MSELoss()
    criterion = SoftDTW(use_cuda=False, gamma=gamma, normalize=True)
    num_of_obs = test_batch_velocity.shape[1]
    dset_train_num = train_traj_abs.shape[1]
    train_traj_norm = train_traj_abs - train_traj_abs[0]
    test_traj_norm = test_traj_abs - test_traj_abs[0]


    for i in range(num_of_obs):
        print(f'Batch: {step} Deal with goal estimation of pedestrian No. {i+1} / {num_of_obs}')
        tmp_traj_v = torch.Tensor(test_batch_velocity[:obs_length,i])
        tmp_traj_v = tmp_traj_v.unsqueeze(0).permute(0,2,1)

        """ replicate all test data and then do loss"""
        tmp_traj_v = tmp_traj_v.repeat(dset_train_num, 1, 1)



        loss_v = criterion(
            tmp_traj_v.permute(0, 2, 1),
            torch.Tensor(train_velocity[:obs_length]).permute(1, 0, 2)
        )

        train_pred_traj_norm = torch.Tensor(train_traj_norm[obs_length:])
        col_pred_traj = train_pred_traj_norm.permute(1, 2, 0)


        """Try the clustering fashion"""
        min_k, min_k_indices = torch.topk(loss_v, 100, largest=False)
        retrieved_expert = col_pred_traj[min_k_indices][:, :, -1]

        # kmeans
        kmeans = KMeans(n_clusters=20, random_state=0).fit(
            retrieved_expert.cpu().numpy()
        )


        """ Find the common between them? """
        """
        Choose use velocity as selecting criterion;
        """

        min_k_end = []
        end_point_appr = []
        for k in kmeans.cluster_centers_:

            # Calculate the absolute end point estimation;
            # test_end = data.pred_traj_gt[:, -1, i].cuda()
            test_end = torch.Tensor(test_traj_norm[-1,i])


            # exp_end = torch.from_numpy(k).cuda()
            exp_end = torch.from_numpy(k)


            min_k_end.append(torch.norm(exp_end - test_end, 2))

            end_point_appr.append(exp_end)

        minimum_loss = min(min_k_end)
        all_min_end.append(minimum_loss)
        print("Min loss of end point estimation is {}".format(all_min_end[-1]))
        predicted_endpoint_norm = end_point_appr[min_k_end.index(minimum_loss)]
        predicted_endpoint = predicted_endpoint_norm + test_traj_abs[0, i]
        predicted_endpoint = predicted_endpoint.unsqueeze(0).numpy()
        rest_diff.append(predicted_endpoint)

    return all_min_end, rest_diff


if __name__ == "__main__":
    pass
