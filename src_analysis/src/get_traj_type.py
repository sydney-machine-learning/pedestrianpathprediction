import numpy as np

from .traj_type_function import *
import torch

def get_type(all_scene, batch_num, test_set_index):
    batch_num = torch.Tensor(batch_num)
    cumsum = torch.cumsum(batch_num, dim=0)
    st_ed = []
    for idx in range(1, cumsum.shape[0]):
        st_ed.append((int(cumsum[idx - 1]), int(cumsum[idx])))

    st_ed.insert(0, (0, int(cumsum[0])))

    all_scene = torch.Tensor(all_scene)
    all_scene = all_scene.permute(1,0,2)

    all_ped_size = all_scene.shape[0]
    scene_size = len(st_ed)
    single_traj_type = np.zeros((all_ped_size, 3))
    if_interaction = np.zeros((scene_size, 1))
    interaction_type = np.zeros((scene_size, 3))

    static_threshold = 1
    linear_threshold = 0.5

    inter_pos_range = 15
    inter_dist_thresh = 5

    grp_dist_thresh = 0.8
    grp_std_thresh = 0.2

    obs_len = 8
    pred_len = 12

    for idx, value in enumerate(st_ed):
        ped_scene_size = batch_num[idx]
        for i in range(value[0], value[1]):
            traj = all_scene[i]
            first_position = traj[0]
            last_position = traj[-1]
            dis_bet_pos = euclidean_distance(first_position, last_position, ped_scene_size)
            if_static = dis_bet_pos < static_threshold
            if if_static:
                single_traj_type[i][0] = 1
                continue

            linear_value = judge_linear(traj, obs_len, pred_len)
            if_linear = linear_value < linear_threshold
            if if_linear:
                single_traj_type[i][1] = 1
                continue

            single_traj_type[i][2] = 1


        if ped_scene_size != 1:
            scene = all_scene[value[0]: value[1]]
            scene = scene.permute(1, 0, 2)
            interaction_matrix = check_interaction(scene, pos_range=inter_pos_range,
                                                   dist_thresh=inter_dist_thresh, obs_len=obs_len)

            interaction_matrix = np.any(interaction_matrix)
            interaction_index = check_group(scene, grp_dist_thresh, grp_std_thresh, obs_len)
            interaction_index = np.any(interaction_index)

            if interaction_matrix or interaction_index:
                if_interaction[idx][0] = 1
                sub_tag = get_interaction_type(scene, inter_pos_range,
                                               inter_dist_thresh, obs_len)[0]

                if sub_tag != 4:
                    interaction_type[idx][sub_tag-1] = 1

    return single_traj_type, if_interaction, interaction_type










