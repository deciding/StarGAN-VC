import os
import argparse
from solver import Solver
from data_loader import get_loader, TestDataset
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    cudnn.benchmark = True

    config.log_dir=os.path.join(config.exp_dir, config.log_dir)
    config.model_save_dir=os.path.join(config.exp_dir, config.model_save_dir)
    config.sample_dir=os.path.join(config.exp_dir, config.sample_dir)
    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    if config.dataset=='vcc2018':
        src_spk='VCC2SM1'
        trg_spk='VCC2SF1'
    else:
        src_spk='p262'
        trg_spk='p272'
    # Data loader.
    train_loader = get_loader(config.train_data_dir, config.batch_size, 'train', num_workers=config.num_workers)
    test_loader = TestDataset(config.test_data_dir, config.wav_dir, src_spk=src_spk, trg_spk=trg_spk) # will convert to train dir...

    # Solver for training and testing StarGAN.
    solver = Solver(train_loader, test_loader, config)

    if config.mode == 'train':
        solver.train()

    # elif config.mode == 'test':
    #     solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cls', type=float, default=10, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--dataset', type=str, default='vcc2018', choices=['vctk', 'vcc2018'])
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    #TODO eliminate
    parser.add_argument('--num_speakers', type=int, default=10, help='dimension of speaker labels')
    parser.add_argument('--train_data_dir', type=str, default='./data/mc/train')
    parser.add_argument('--test_data_dir', type=str, default='./data/mc/test')
    parser.add_argument('--wav_dir', type=str, default="./data/VCTK-Corpus/wav16")

    parser.add_argument('--exp_dir', type=str, default="exp")
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--sample_dir', type=str, default='./samples')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    if config.dataset=='vcc2018':
        config.train_data_dir='vcc2018/mc_vcc2018/train'
        config.test_data_dir='vcc2018/mc_vcc2018/test'
        config.wav_dir='vcc2018/vcc2018_evaluation_16k'
        config.num_speakers=4
    else:
        config.train_data_dir='data/mc/train'
        config.test_data_dir='data/mc/test'
        config.wav_dir='data/VCTK-Corpus/wav16'
        config.num_speakers=10

    print(config)
    main(config)
