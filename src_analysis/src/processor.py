import torch
import torch.nn as nn

from .dynamics_based import DYNA
from .utils import *

from tqdm import tqdm
from .traj_report import *
from .store_and_load import *
from .calculate_timestep_gradient import *

class processor(object):
    def __init__(self, args):

        self.args = args

        self.dataloader = Trajectory_Dataloader(args)
        self.net = DYNA(args)

        self.set_optimizer()
        if self.args.using_cuda:
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()

        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1

    def save_model(self, epoch):

        model_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                     str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)

    def load_model(self):

        if self.args.load_model is not None:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.tar'
            print(self.args.model_save_path)
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

    def test(self):

        print('Testing begin')
        self.load_model()
        self.net.eval()
        if self.args.only_show_result:
            self.only_show_results()
        else:
            if self.args.calculate_gradient:
                calculate_time_gradient(self.net, self.dataloader, self.args)
            test_error, test_final_error = self.test_epoch()
            print('Set: {}, epoch: {},test_error: {} test_final_error: {}'.format(self.args.test_set,
                                                                                          self.args.load_model,
                                                                                       test_error, test_final_error))

    @torch.no_grad()
    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5

        all_best_pred_traj = {'best_pred_traj': []}
        all_traj_scene_type = {'traj_scene_type': []}
        all_batch_pednum = {'batch_pednum': []}
        all_metric = {'ade':[], 'fde':[]}
        test_set = ['eth', 'hotel', 'zara1', 'zara2', 'univ']

        for batch in tqdm(range(self.dataloader.testbatchnums)):

            inputs, batch_id, traj_scene_type = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum, batch_velocity = inputs

            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                              :-1], batch_pednum

            all_output = []
            for i in range(self.args.sample_num):
                outputs_infer = self.net.forward(inputs_forward, iftest=True)
                all_output.append(outputs_infer)
            self.net.zero_grad()

            all_output = torch.stack(all_output)

            lossmask, num = getLossMask(all_output, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)

            error, error_cnt, final_error, final_error_cnt, best_predicted_traj, ade, fde = L2forTestS(all_output, batch_abs[1:, :, :2],
                                                                        self.args.obs_length, lossmask)

            all_best_pred_traj['best_pred_traj'].append(best_predicted_traj)
            all_traj_scene_type['traj_scene_type'].append(traj_scene_type)
            all_batch_pednum['batch_pednum'].append(batch_pednum)
            all_metric['ade'].append(ade)
            all_metric['fde'].append(fde)


            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        test_set_name = test_set[self.args.test_set]
        save_directory = self.args.save_dir
        report_file = open(os.path.join(save_directory, f'{test_set_name}_XAI_report.txt'), 'a+')
        write_traj_scene_result(all_best_pred_traj, all_traj_scene_type, all_batch_pednum, all_metric, report_file)
        report_file.write(f'All Mean ADE = {error_epoch / error_cnt_epoch}; ALL Mean FDE = {final_error_epoch / final_error_cnt_epoch}\n')
        report_file.close()
        store_important_data(all_best_pred_traj, all_traj_scene_type, all_batch_pednum, all_metric, save_directory, test_set_name)

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch

    def only_show_results(self):
        test_set = ['eth', 'hotel', 'zara1', 'zara2', 'univ']
        test_set_name = test_set[self.args.test_set]
        save_directory = self.args.save_dir
        report_file = open(os.path.join(save_directory, f'{test_set_name}_XAI_report.txt'), 'a+')
        all_batch_pednum, all_best_pred_traj, all_traj_scene_type, all_metric = load_important_data(save_directory, test_set_name)
        write_traj_scene_result(all_best_pred_traj, all_traj_scene_type, all_batch_pednum, all_metric, report_file)
        report_file.close()
