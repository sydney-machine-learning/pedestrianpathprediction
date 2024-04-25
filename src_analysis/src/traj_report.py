import torch
from tqdm import tqdm
import os

def write_traj_scene_result(all_best_pred_traj, all_traj_scene_type, all_batch_pednum, all_metric, report_file):
    all_batch_pednum = all_batch_pednum['batch_pednum']
    all_best_pred_traj = all_best_pred_traj['best_pred_traj']
    all_traj_scene_type = all_traj_scene_type['traj_scene_type']
    all_ade = all_metric['ade']
    all_fde = all_metric['fde']
    batch_size = len(all_batch_pednum)

    # for the trajectory
    linear_traj = {'ade':[], 'fde': []}
    num_linear_traj = 0
    static_traj = {'ade':[], 'fde': []}
    num_static_traj = 0
    other_traj = {'ade':[], 'fde': []}
    num_other_traj = 0

    # for the scene (interaction)
    leader_follower = {'ade':[], 'fde': []}
    num_traj_leader_follower = 0
    collision_avoidance = {'ade':[], 'fde': []}
    num_traj_collision_avoidance = 0
    group = {'ade':[], 'fde': []}
    num_traj_group = 0
    other_interaction = {'ade':[], 'fde': []}
    num_other_interaction = 0

    for i in tqdm(range(batch_size)):
        traj_scene_type = all_traj_scene_type[i]
        single_traj_type, if_interaction, interaction_type = traj_scene_type
        ADE = all_ade[i]
        FDE = all_fde[i]
        batch_num = all_batch_pednum[i]
        cumsum = torch.cumsum(batch_num, dim=0)
        st_ed = []
        for idx in range(1, cumsum.shape[0]):
            st_ed.append((int(cumsum[idx - 1]), int(cumsum[idx])))
        st_ed.insert(0, (0, int(cumsum[0])))


        for idx, value in enumerate(st_ed):
            ped_scene_size = batch_num[idx]
            for j in range(value[0], value[1]):
                if single_traj_type[j][0]:
                    linear_traj['ade'].append(ADE[j])
                    linear_traj['fde'].append(FDE[j])
                    num_linear_traj += 1
                elif single_traj_type[j][1]:
                    static_traj['ade'].append(ADE[j])
                    static_traj['fde'].append(FDE[j])
                    num_static_traj += 1
                elif single_traj_type[j][2]:
                    other_traj['ade'].append(ADE[j])
                    other_traj['fde'].append(FDE[j])
                    num_other_traj += 1

            if if_interaction[idx]:
                if interaction_type[idx][0]:
                    leader_follower_ade = ADE[value[0]:value[1]]
                    leader_follower_fde = FDE[value[0]:value[1]]
                    num_traj_leader_follower += ped_scene_size
                    leader_follower['ade'].append(leader_follower_ade.sum())
                    leader_follower['fde'].append(leader_follower_fde.sum())
                elif interaction_type[idx][1]:
                    collision_avoidance_ade = ADE[value[0]:value[1]]
                    collision_avoidance_fde = FDE[value[0]:value[1]]
                    num_traj_collision_avoidance += ped_scene_size
                    collision_avoidance['ade'].append(collision_avoidance_ade.sum())
                    collision_avoidance['fde'].append(collision_avoidance_fde.sum())
                elif interaction_type[idx][2]:
                    group_ade = ADE[value[0]:value[1]]
                    group_fde = FDE[value[0]:value[1]]
                    num_traj_group += ped_scene_size
                    group['ade'].append(group_ade.sum())
                    group['fde'].append(group_fde.sum())
            else:
                other_interaction_ade = ADE[value[0]:value[1]]
                other_interaction_fde = FDE[value[0]:value[1]]
                num_other_interaction += ped_scene_size
                other_interaction['ade'].append(other_interaction_ade.sum())
                other_interaction['fde'].append(other_interaction_fde.sum())

    linear_traj_mean_ade = sum(linear_traj['ade'])/ len(linear_traj['ade'])
    linear_traj_mean_fde = sum(linear_traj['fde'])/ len(linear_traj['fde'])
    report_file.write(f'For linear trajectory, Mean ADE = {linear_traj_mean_ade}; Mean FDE = {linear_traj_mean_fde}'
                      f' Number of trajectories: {num_linear_traj}\n')
    print(f'For linear trajectory, Mean ADE = {linear_traj_mean_ade}; Mean FDE = {linear_traj_mean_fde}'
          f' Number of trajectories: {num_linear_traj}')

    static_traj_mean_ade = sum(static_traj['ade']) / len(static_traj['ade'])
    static_traj_mean_fde = sum(static_traj['fde']) / len(static_traj['fde'])
    report_file.write(f'For static trajectory, Mean ADE = {static_traj_mean_ade}; Mean FDE = {static_traj_mean_fde}'
                      f' Number of trajectories: {num_static_traj}\n')
    print(f'For static trajectory, Mean ADE = {static_traj_mean_ade}; Mean FDE = {static_traj_mean_fde}'
          f' Number of trajectories: {num_static_traj}')

    other_traj_mean_ade = sum(other_traj['ade']) / len(other_traj['ade'])
    other_traj_mean_fde = sum(other_traj['fde']) / len(other_traj['fde'])
    report_file.write(f'For other classified trajectory, Mean ADE = {other_traj_mean_ade}; Mean FDE = {other_traj_mean_fde}'
                      f' Number of trajectories: {num_other_traj}\n')
    print(f'For other classified trajectory, Mean ADE = {other_traj_mean_ade}; Mean FDE = {other_traj_mean_fde}'
          f' Number of trajectories: {num_other_traj}')

    leader_follower_mean_ade = sum(leader_follower['ade']) / num_traj_leader_follower
    leader_follower_mean_fde = sum(leader_follower['fde']) / num_traj_leader_follower
    report_file.write(
        f'For leader follower scene, Mean ADE = {leader_follower_mean_ade}; Mean FDE = {leader_follower_mean_fde}; '
        f' Number of scenes: {num_traj_leader_follower}\n')
    print(
        f'For leader follower scene, Mean ADE = {leader_follower_mean_ade}; Mean FDE = {leader_follower_mean_fde}'
        f' Number of scenes: {num_traj_leader_follower}')

    collision_avoidance_mean_ade = sum(collision_avoidance['ade']) / num_traj_collision_avoidance
    collision_avoidance_mean_fde = sum(collision_avoidance['fde']) / num_traj_collision_avoidance
    report_file.write(
        f'For collision avoidance scene, Mean ADE = {collision_avoidance_mean_ade}; Mean FDE = {collision_avoidance_mean_fde}'
        f' Number of scenes: {num_traj_collision_avoidance}\n')
    print(
        f'For collision avoidance scene, Mean ADE = {collision_avoidance_mean_ade}; Mean FDE = {collision_avoidance_mean_fde}'
        f' Number of scenes: {num_traj_collision_avoidance}')

    group_mean_ade = sum(group['ade']) / num_traj_group
    group_mean_fde = sum(group['fde']) / num_traj_group
    report_file.write(
        f'For group scene, Mean ADE = {group_mean_ade}; Mean FDE = {group_mean_fde}'
        f' Number of scenes: {num_traj_group}\n')
    print(
        f'For group scene, Mean ADE = {group_mean_ade}; Mean FDE = {group_mean_fde}'
        f' Number of scenes: {num_traj_group}')

    other_interaction_mean_ade = sum(other_interaction['ade']) / num_other_interaction
    other_interaction_mean_fde = sum(other_interaction['fde']) / num_other_interaction
    report_file.write(
        f'For other interaction scene, Mean ADE = {other_interaction_mean_ade}; Mean FDE = {other_interaction_mean_fde}'
        f' Number of scenes: {num_other_interaction}\n')
    print(
        f'For other interaction scene, Mean ADE = {other_interaction_mean_ade}; Mean FDE = {other_interaction_mean_fde}'
        f' Number of scenes: {num_other_interaction}')

