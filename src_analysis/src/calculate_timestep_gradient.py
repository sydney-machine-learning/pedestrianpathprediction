from tqdm import tqdm
import torch
import os

def calculate_time_gradient(network, dataloader, args):
    dataloader.reset_batch_pointer(set='test')
    error_epoch, final_error_epoch = 0, 0,
    error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5

    gradient_norms = []
    for i in range(args.obs_length):
        gradient_norms.append([])

    for batch in tqdm(range(dataloader.testbatchnums)):
        inputs, batch_id, _ = dataloader.get_test_batch(batch)
        inputs = tuple([torch.Tensor(i) for i in inputs])

        if args.using_cuda:
            inputs = tuple([i.cuda() for i in inputs])

        batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum, batch_velocity = inputs

        inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                          :-1], batch_pednum
        all_timesteps = inputs_forward[0].shape[0]
        gradient_norms_per_timestep = []

        inputs_forward[0].requires_grad = True
        inputs_forward[1].requires_grad = True


        outputs_infer = network.forward(inputs_forward, iftest=True)

        network.zero_grad()
        if inputs_forward[0].grad is not None:
            inputs_forward[0].grad.data.zero_()
        if inputs_forward[1].grad is not None:
            inputs_forward[1].grad.data.zero_()

        outputs_infer[args.obs_length - 1:].backward(torch.ones_like(outputs_infer[args.obs_length - 1:]),
                                                     retain_graph=True)

        for t in range(args.obs_length):
            gradient_norm = inputs_forward[0].grad.norm(dim=2)[t]
            gradient_norms[t].append(gradient_norm)

    test_set = ['eth', 'hotel', 'zara1', 'zara2', 'univ']
    test_set_name = test_set[args.test_set]
    save_directory = args.save_dir
    report_file = open(os.path.join(save_directory, f'{test_set_name}_gradient_report.txt'), 'a+')

    summed_gradient_norm = []
    for t in range(args.obs_length):
        gradient_norm_t = gradient_norms[t]
        summed_gradient_norm_t = 0
        for i in range(len(gradient_norm_t)):
            summed_gradient_norm_t += gradient_norm_t[i].sum(dim=0).item()
        summed_gradient_norm.append(summed_gradient_norm_t)

    normalized_gradient_norms =[x / sum(summed_gradient_norm) for x in summed_gradient_norm]
    report_file.write(f'the correlation of {test_set_name} in motion history with respect to predicted trajectory is\n'
                      f'{normalized_gradient_norms}')
    report_file.close()





