import os
import pickle

def store_important_data(all_best_pred_traj, all_traj_scene_type, all_batch_pednum, all_metric, save_directory,
                     test_set_name):
    batch_file_path = os.path.join(save_directory, f'batch_pednum_{test_set_name}.pkl')
    with open(batch_file_path, 'wb') as f:
        pickle.dump(all_batch_pednum, f)

    best_traj_file_path = os.path.join(save_directory, f'best_predicted_trajectory_{test_set_name}.pkl')
    with open(best_traj_file_path, 'wb') as f:
        pickle.dump(all_best_pred_traj, f)

    traj_scene_type_path = os.path.join(save_directory, f'traj_scene_type_{test_set_name}.pkl')
    with open(traj_scene_type_path, 'wb') as f:
        pickle.dump(all_traj_scene_type, f)

    metric_path = os.path.join(save_directory, f'metric_{test_set_name}.pkl')
    with open(metric_path, 'wb') as f:
        pickle.dump(all_metric, f)

    print('Save done.')

def load_important_data(save_directory, test_set_name):
    batch_file_path = os.path.join(save_directory, f'batch_pednum_{test_set_name}.pkl')
    with open(batch_file_path, 'rb') as f:
        all_batch_pednum = pickle.load(f)

    best_traj_file_path = os.path.join(save_directory, f'best_predicted_trajectory_{test_set_name}.pkl')
    with open(best_traj_file_path, 'rb') as f:
        all_best_pred_traj = pickle.load(f)

    traj_scene_type_path = os.path.join(save_directory, f'traj_scene_type_{test_set_name}.pkl')
    with open(traj_scene_type_path, 'rb') as f:
        all_traj_scene_type = pickle.load(f)

    metric_path = os.path.join(save_directory, f'metric_{test_set_name}.pkl')
    with open(metric_path, 'rb') as f:
        all_metric = pickle.load(f)

    print('Load done.')
    return all_batch_pednum, all_best_pred_traj, all_traj_scene_type, all_metric





