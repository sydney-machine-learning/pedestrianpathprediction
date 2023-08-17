import os
from src.helper_expert import *
import numpy as np
import pickle

def goal_estimator(test_set_name, dataloader):
    save_directory = "./goal_estimated_result/eth_ucy/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    obs_length = dataloader.args.obs_length
    pred_length = dataloader.args.pred_length
    seq_length = obs_length + pred_length

    trainset = dataloader.trainbatch
    train_batch_num = len(trainset)
    train_traj_abs = np.zeros((seq_length, 0, 2))
    train_traj_v = np.zeros((seq_length, 0, 2))

    for i in range(train_batch_num):
        train_traj_batch = trainset[i][0][0]
        train_velocity_batch = trainset[i][0][5]
        train_traj_abs = np.append(train_traj_abs, train_traj_batch, 1)
        train_traj_v = np.append(train_traj_v, train_velocity_batch, 1)


    testset = dataloader.testbatch
    test_batch_num = len(testset)


    estimated_goal_result = {'Predicted_Goal': [], 'True_Goal': []}
    estimated_goal_error = {'Estimated_Goal_Error': []}
    step = 0
    for i in range(test_batch_num):
        test_batch_velocity = testset[i][0][5]
        test_traj_abs = testset[i][0][0]
        step += 1
        end_error, test_predicted_end = expert_find(test_batch_velocity, test_traj_abs, train_traj_v, train_traj_abs, obs_length, step, gamma=1.0)
        if len(test_predicted_end) == 0:
            continue
        test_predicted_end = np.concatenate(test_predicted_end, 0)
        test_real_end = test_traj_abs[-1]
        estimated_goal_error['Estimated_Goal_Error'].append(end_error)
        estimated_goal_result['Predicted_Goal'].append(test_predicted_end)
        estimated_goal_result['True_Goal'].append(test_real_end)

    print('Save Model...')
    with open(os.path.join(save_directory, f'goal_estimated_{test_set_name}.pkl'), 'wb') as f:
        pickle.dump(estimated_goal_result, f)

    print('Save error...')
    with open(os.path.join(save_directory, f'goal_estimated_error_{test_set_name}.pkl'), 'wb') as f:
        pickle.dump(estimated_goal_error, f)


