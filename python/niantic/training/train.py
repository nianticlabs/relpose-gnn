"""
Example call:
   python -u ${RELPOSEGNN}/python/niantic/training/train.py \
    --dataset-dir "${SEVENSCENES}" \
    --train-data-dir "${SEVENSCENESRW}" \
    --test-data-dir "${SEVENSCENESRW}" \
    --save-dir "${DATADIR}" \
    --gpu 0 \
    --experiment 0 \
    --test-scene multi
"""
import argparse
import os.path as osp
import sys
from pathlib import Path
import random

import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, Subset, ConcatDataset
from torch_geometric.loader import DataLoader
from torchvision import models
from tqdm import tqdm

#extend PYTHONPATH
p_parent = Path(__file__).parent.resolve()
p_python = str(Path(*p_parent.parts[:p_parent.parts.index('niantic')]))
if p_python not in sys.path:
    sys.path.insert(0, p_python)

from niantic.datasets.dataset_7Scenes_multi import SEVEN_SCENES_multi
from niantic.modules.criterion import PoseNetCriterion
from niantic.modules.posenet import PoseNetX_LIGHT_KNN, PoseNetX_R2
from niantic.utils.utils import save_checkpoint
from niantic.utils.pose_utils import quaternion_angular_error, qexp


class MultiModelTrainer:
    def __init__(self, args):
        # ---------Hyperparameters-------
        self.experiment = args.experiment  # normal or leave-one-out
        self.dataset = args.dataset  # '7Scenes'
        self.train_scene = args.train_scene  # 'multi'
        self.test_scene = args.test_scene # heads
        self.train_data_dir = args.train_data_dir
        self.test_data_dir = args.test_data_dir
        self.save_dir = args.save_dir
        self.model_name = args.model_name
        self.save_model = True # save model parameters?
        self.img_H = 256  # input image height
        self.batch_size = 8
        self.graph_structure = 'fc'
        self.test_graph_structure = 'fc'
        self.seq_len = 8
        self.test_seq_len = 8
        self.use_VO_loss = True
        self.use_VO_model = True
        self.lr = 5e-5  # initial LR
        self.lr_decay = 0.1 # learning rate decay factor le:=lr/lr_decay
        self.lr_decay_step = 20 # number of batch for decay
        self.weight_decay = 0.0005 # L2 reg
        self.sax = 0.0  # initial absolute translation coeff
        self.saq = args.saq  # initial absolute rotation coeff
        self.srx = 0.0  # initial relative translation coeff
        self.srq = args.srq  # initial relative rotation coeff
        self.learn_gamma = True # gamma is learnable or fixed
        self.lambda_AP = args.lambda_AP  # 0.0
        self.edge_keep_factor = 0.5 # Edge dropout factor during training
        self.gnn_recursion = args.gnn_recursion # number of recursion for GNN: 2, 3
        self.exp_name = args.exp_name
        self.weights_filename = args.weights_filename # weight filename if pretrained model provided
        self.seed = args.seed
        self.num_workers = args.num_workers
        # -------------------------------

        # Device - GPU, CPU
        self.device = f'cuda:{args.gpu}' \
            if args.gpu is not None \
            else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Define datasets and dataloaders
        #Training datasets
        if self.experiment == 0 and args.dataset == '7Scenes':
            self.training_scenes = ['heads','chess','redkitchen','pumpkin','office', 'fire','stairs']
        elif self.experiment == 1 and args.dataset == '7Scenes':
            self.training_scenes = ['heads','chess','redkitchen','pumpkin','office', 'fire','stairs']
            self.training_scenes.remove(args.test_scene)
        elif self.experiment == 2 and args.dataset == '7Scenes':
            self.training_scenes = [args.train_scene]

        #Test datasets
        if args.test_scene == 'multi' and args.dataset == '7Scenes':
            self.test_scenes = ['heads','chess','redkitchen','pumpkin','office', 'fire','stairs']
        elif args.dataset == '7Scenes':
            self.test_scenes = [args.test_scene]

        train_dataset_list = list()
        test_dataset_list = list()

        for s in self.training_scenes:
            train_data_file = args.train_data_dir + s + '_fc8_sp5_train'
            train_dataset_list.append( SEVEN_SCENES_multi(
                root=f'{train_data_file}',
                seqs=[], train=True, database_set='train', seq_len=self.seq_len,
                graph_structure='fc', excluded_scenes=None, device_id=args.gpu))

        for s in self.test_scenes:
            test_data_file = args.test_data_dir + s + '_fc8_sp5_test'
            test_dataset_list.append( SEVEN_SCENES_multi(
                root=f'{test_data_file}',
                seqs=[s], train=False, database_set='train', seq_len=self.test_seq_len,
                graph_structure='fc', device_id=args.gpu) )

        self.train_dataset = ConcatDataset(train_dataset_list)
        self.test_dataset_list = test_dataset_list


        self.train_loader = \
            DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                       num_workers=args.num_workers)

        x = self.train_dataset[0].x
        self.num_nodes, self.feat_dim = x.size()
        self.img_W = int(self.feat_dim / (3 * self.img_H))
        self.max_num_iter = len(self.train_dataset) // self.batch_size

        logger.info(f'Experiment name: {self.exp_name}')
        logger.info(f'Dataset: {self.dataset}')
        logger.info(f'Train scene: {self.train_scene}')
        logger.info(f'Test scene: {self.test_scene}')
        logger.info(f'Train data dir: {self.train_data_dir}')
        logger.info(f'Test data dir: {self.test_data_dir}')
        logger.info(f'Train dataset size: {len(self.train_dataset)}')
        logger.info(f'batch_size: {self.batch_size}')
        logger.info(f'Images sizes: {self.img_H}, {self.img_W}')
        logger.info(f'Number of nodes in the graph: {self.num_nodes}, {self.graph_structure}')
        logger.info(f'Number of nodes in the graph - test: '
                    f'{self.test_seq_len} {self.test_graph_structure}')
        logger.info(f'Model: {self.model_name}')
        logger.info(f'Use RP loss: {self.use_VO_loss}')
        logger.info(f'Use RP model: {self.use_VO_model}')
        logger.info(f'lambda_AP: {self.lambda_AP}')
        logger.info(f'srx: {self.srx}')
        logger.info(f'srq: {self.srq}')
        logger.info(f'edge_keep_factor: {self.edge_keep_factor}')
        logger.info(f'gnn_recursion: {self.gnn_recursion}')
        logger.info(f'droprate: {args.droprate}')
        logger.info(f'gpu: {args.gpu}')
        logger.info(f'seed: {args.seed}')

        # Define model
        self.feature_extractor = models.resnet34(pretrained=True)
        if self.model_name == 'R1' or self.model_name == 'light_knn':
            self.model = PoseNetX_LIGHT_KNN(
                feature_extractor=self.feature_extractor, droprate=args.droprate, pretrained=True,
                use_gnn=self.graph_structure != 'ind', knn=-1).to(self.device)
        elif self.model_name == 'R2':
            self.model = PoseNetX_R2(
                feature_extractor=self.feature_extractor, droprate=args.droprate, pretrained=True,
                use_gnn=self.graph_structure != 'ind', knn=-1, gnn_recursion=self.gnn_recursion) \
                .to(self.device)
        elif self.model_name == 'R3':
            self.model = PoseNetX_R2(
                self.feature_extractor, droprate=args.droprate, pretrained=True,
                use_gnn=self.graph_structure != 'ind',
                knn=-1, gnn_recursion=self.gnn_recursion, feat_dim=2048, edge_feat_dim=2048,
                node_dim=2048) \
                .to(self.device)

        self.model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.params = sum([np.prod(p.size()) for p in self.model_parameters])
        logger.info(f'Num parameters: {self.params}')

        # Define loss
        self.train_criterion = \
            PoseNetCriterion(sax=self.sax, saq=self.saq, learn_beta=True).to(self.device)
        self.train_criterion_R = \
            PoseNetCriterion(sax=self.srx, saq=self.srq, learn_beta=True).to(self.device)
        self.val_criterion = PoseNetCriterion().to(self.device)

        # Define optimizer
        param_list = [{'params': self.model.parameters()}]
        if self.learn_gamma and hasattr(self.train_criterion, 'sax') \
          and hasattr(self.train_criterion, 'saq'):
            param_list.append({'params': [self.train_criterion.sax, self.train_criterion.saq]})
        if self.learn_gamma and hasattr(self.train_criterion_R, 'sax') \
          and hasattr(self.train_criterion_R, 'saq'):
            param_list.append({'params': [self.train_criterion_R.sax, self.train_criterion_R.saq]})

        self.optimizer = torch.optim.Adam(param_list, lr=self.lr, weight_decay=self.weight_decay)

        #Load weights if provided
        if osp.isfile(self.weights_filename):
          checkpoint = torch.load(self.weights_filename)
          self.model.load_state_dict(checkpoint['model_state_dict'])
          logger.info('Loaded weights from {:s}'.format(self.weights_filename))
        else:
          logger.info('Could not load weights from {:s}'.format(self.weights_filename))


    def train(self, epoch: int):
        self.model.train()
        if epoch > 1 and epoch % self.lr_decay_step == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.lr_decay
                print('LR: ', param_group['lr'])

        for batch_idx, data in tqdm(enumerate(self.train_loader),
                                    desc=f'[Epoch {epoch:04d}] train',
                                    total=len(self.train_loader)):

            if batch_idx == self.max_num_iter:
                break

            # Data augmentation --------------------------------
            # Randomly remove some edges
            num_edges = int(data.edge_index.shape[1] // (self.batch_size * 2))
            surviving_edges = np.random.random(num_edges) < self.edge_keep_factor
            if np.sum(surviving_edges) == 0:
                surviving_edges = surviving_edges + 1
            surviving_edges = np.tile(surviving_edges, 2 * self.batch_size)
            surviving_edges = surviving_edges.tolist()
            # if len(surviving_edges) <data.edge_index.shape[1]:
            #     data.edge_index = data.edge_index[:,surviving_edges]
            data.edge_attr = data.edge_attr[surviving_edges, :]
            # ----------------------------------------------------

            target = data.y
            target = target.to(self.device)

            self.optimizer.zero_grad()

            if self.use_VO_loss:
                pred, pred_R, edge_index = self.model(data.to(self.device))

                target_R = self.model.compute_RP(target, edge_index)

                loss_R = self.train_criterion_R(
                    pred_R.view(1, pred_R.size(0), pred_R.size(1)),
                    target_R.view(1, target_R.size(0), target_R.size(1)))

                loss_total = loss_R[0]
            else:
                pred = self.model(data.to(self.device))

                loss = self.train_criterion(pred.view(1, pred.size(0), pred.size(1)),
                                            target.view(1, target.size(0), target.size(1)))

                loss_total = loss[0]

            loss_total.backward()
            self.optimizer.step()


    def eval_RP(self, data_set, ref_node, epoch: int = None, set='test', scene='heads'):
        self.model.eval()

        L = len(data_set)  # * T
        loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        pred_poses = np.zeros((L, 7))  # store all predicted poses
        targ_poses = np.zeros((L, 7))  # store all target poses

        # loss functions
        t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        q_criterion = quaternion_angular_error

        # inference loop
        for batch_idx, data in tqdm(enumerate(loader), desc=f'[Epoch {epoch:04d}] eval',
                                    total=len(loader)):

            # output : 1 x 6 or 1 x STEPS x 6
            output, output_R, edge_index = self.model(data.to(self.device))

            s = output.size()
            output_R = output_R.cpu().data.numpy().reshape((-1, s[-1]))

            target = data.y
            target = target.to('cpu').numpy().reshape((-1, s[-1]))

            edges = edge_index.cpu().data.numpy()

            # Choose one reference absoulte pose and compute the absolute poses in the subgraph
            # using predicted relative poses
            valid_edges = edges[1] == 0

            ref_idx = np.argwhere(valid_edges)[ref_node, 0]
            RP_estimate = output_R[ref_idx, :]
            reference_AP = target[edges[0, ref_idx], :]
            output = reference_AP - RP_estimate
            output = np.expand_dims(output, axis=0)

            # normalize the predicted quaternions
            q = [qexp(p[3:]) for p in output]
            output = np.hstack((output[:, :3], np.asarray(q)))
            q = [qexp(p[3:]) for p in target]
            target = np.hstack((target[:, :3], np.asarray(q)))

            # take the first prediction
            pred_poses[batch_idx, :] = output[0]
            targ_poses[batch_idx, :] = target[0]

        # calculate losses
        t_loss = np.asarray([t_criterion(p, t)
                             for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
        q_loss = np.asarray([q_criterion(p, t)
                             for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])

        median_t = np.median(t_loss)
        median_q = np.median(q_loss)
        mean_t = np.mean(t_loss)
        mean_q = np.mean(q_loss)

        logger.info(f'[Scene: {scene}, set: {set}, Epoch {epoch:04d}] Error in translation:'
                    f' median {median_t:3.2f} m,'
                    f' mean {mean_t:3.2f} m'
                    f'\tError in rotation:'
                    f' median {median_q:3.2f} degrees,'
                    f' mean {mean_q:3.2f} degrees')
        return median_t, mean_t, median_q, mean_q


def _parse_args(argv):
    parser = argparse.ArgumentParser('')
    parser.add_argument('--experiment', type=int,
                        help='Type of experiment (multi-scene training:0, leave-one-out:1, single-scene training:2)', default=0)
    parser.add_argument('--dataset', type=str, help='Name of dataset', default='7Scenes')
    parser.add_argument('--train-scene', type=str, help='Name of sequence used for training', default='multi')
    parser.add_argument('--test-scene', type=str, help='Which scene to test on', default='multi')
    parser.add_argument('--train-data-dir', type=str, help='Path to train/test data',
                        default='/mnt/data-7scenes-ozgur/3dv/data/seven_scenes/')
    parser.add_argument('--test-data-dir', type=str, help='Path to train/test data',
                        default='/mnt/data-7scenes-ozgur/3dv/data/seven_scenes/')
    parser.add_argument('--save-dir', type=str, help='Path to output data', default='/mnt/data-7scenes-ozgur/mozgur/3dv')
    parser.add_argument('--save-model', type=bool, help='Wheather a checkpoint is saved or not', default=True)
    parser.add_argument('--weights-filename', type=str, help='Weight file name for pre-trained model', default='')
    parser.add_argument('--model-name', type=str, help='Name of the model (R1, R2, R3)', default='R3')
    parser.add_argument('--srq', type=int, help='Relative rotation loss weight coefficient', default=-3)
    parser.add_argument('--saq', type=int, help='Absolute rotation loss weight coefficient', default=-3)
    parser.add_argument('--droprate', type=float, help='Droprate', default=0.5)
    parser.add_argument('--gnn-recursion', type=int, help='Number of GNN layers', default=3)
    parser.add_argument('--lambda-AP', dest='lambda_AP', type=float, help='Absolute pose weight coeff', default=0.0)
    parser.add_argument('--max-epoch', type=int, help='Number of epochs', default=41)
    parser.add_argument('--num-workers', type=int, help='Number of dataloader workers', default=8)
    parser.add_argument('--gpu', default=None, help='GpuId', type=int)
    parser.add_argument('--exp-name', default=None, help='experiment name', type=str)
    parser.add_argument('--seed', default=999, help='random seed', type=int)

    args = parser.parse_args(argv)
    if not hasattr(args, 'saq') or args.saq is None:
        setattr(args, 'saq', args.srq)
    if not hasattr(args, 'exp_name') or args.exp_name is None:
        if args.experiment == 2:
            setattr(args, 'exp_name', f'single_w_{args.train_scene}_w_test_{args.test_scene}_seed_{str(args.seed)}')
        elif args.experiment == 1:
            setattr(args, 'exp_name', f'multi_wo_{args.test_scene}_w_test_{args.test_scene}_seed_{str(args.seed)}')
        else:
            setattr(args, 'exp_name', f'multi_w_test_{args.test_scene}_seed_{str(args.seed)}')

    return args


def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------MAIN----------------------
def main(argv, metrics_callback=None):
    args = _parse_args(argv)

    seed_everything(args.seed)
    print(f'Seed: {torch.initial_seed()}')

    model_multi_trainer = MultiModelTrainer(args=args)
    print('TRAINING...')
    best_median_ts = [1e6] * len(model_multi_trainer.test_dataset_list)
    best_median_qs = [1e6] * len(model_multi_trainer.test_dataset_list)
    logdir = Path(args.save_dir) / args.dataset / args.train_scene / args.exp_name
    logger.add(str(logdir / 'logger.log'))
    print('Logger file path: ', str(logdir / 'logger.log'))

    for epoch in tqdm(range(args.max_epoch), desc='epoch', total=args.max_epoch):
        model_multi_trainer.train(epoch=epoch)

        if epoch > 20:
            if args.save_model:
                save_checkpoint(
                    logdir=str(logdir),
                    epoch=epoch, model=model_multi_trainer.model,
                    optimizer=model_multi_trainer.optimizer,
                    train_criterion=model_multi_trainer.train_criterion)

            for j in range(len(model_multi_trainer.test_dataset_list)):
                cur_median_t, _, cur_median_q, _ = model_multi_trainer.eval_RP(
                    data_set=model_multi_trainer.test_dataset_list[j],
                    ref_node=0, epoch=epoch,
                    scene=model_multi_trainer.test_scenes[j])

                if cur_median_t < best_median_ts[j]:
                    best_median_ts[j] = cur_median_t
                if cur_median_q < best_median_qs[j]:
                    best_median_qs[j] = cur_median_q

    return best_median_ts, best_median_qs


if __name__ == '__main__':
    main(sys.argv[1:])
