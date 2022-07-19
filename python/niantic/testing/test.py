"""
Example call:
   python -u ${RELPOSEGNN}/python/niantic/testing/test.py \
     --dataset-dir "${SEVENSCENES}" \
     --test-data-dir "${SEVENSCENESRW}" \
     --weights "${DATADIR}/relpose_gnn__multi_39.pth.tar" \
     --save-dir "${DATADIR}" \
     --gpu 0 \
     --test-scene "${SCENE}"
"""
import argparse
import random
import re
import sys
from pathlib import Path
import os.path as osp

import numpy as np
import torch
from loguru import logger
from torch_geometric.data import DataLoader
from torchvision import models
from tqdm import tqdm

# extend PYTHONPATH
p_parent = Path(__file__).parent.resolve()
p_python = str(Path(*p_parent.parts[:p_parent.parts.index('niantic')]))
if p_python not in sys.path:
    sys.path.insert(0, p_python)

from niantic.datasets.dataset_7Scenes_multi import SEVEN_SCENES_multi
from niantic.datasets.dataset_Cambridge_multi import CAMBRIDGE_multi
from niantic.modules.criterion import PoseNetCriterion
from niantic.modules.posenet import PoseNetX_LIGHT_KNN, PoseNetX_R2
from niantic.utils.pose_utils import quaternion_angular_error, qexp


def save_poses(pred_poses: np.ndarray, rel_paths: list, p_output: Path, target_poses: np.ndarray):
    logger.info(f'Saving to {p_output}')
    assert len(rel_paths) == len(pred_poses), f'len(rel_paths): {len(rel_paths)} != {len(pred_poses)} len(pred_poses)'
    np.savez(p_output, rel_path=rel_paths, abs_t=pred_poses[:, :3], abs_q=pred_poses[:, 3:],
             targ_t=target_poses[:, :3], targ_q=target_poses[:, 3:])


class MultiModelTrainer:
    def __init__(self, args):
        # ---------Hyperparameters-------
        self.dataset = args.dataset  # e.g '7scenes'
        self.test_scene = args.test_scene  # e.g  'heads'
        self.test_data_dir = args.test_data_dir
        assert Path(self.test_data_dir).exists(), self.test_data_dir
        if hasattr(args, 'dataset_dir'):
            self.dataset_dir = args.dataset_dir
        self.save_dir = args.save_dir
        self.model_name = args.model_name
        self.img_H = 256  # input image height
        # self.batch_size = 8
        self.graph_structure = 'fc'
        self.test_graph_structure = 'fc'
        self.seq_len = 8
        self.test_seq_len = 8
        self.use_VO_loss = True
        self.use_VO_model = True
        self.lr = 5e-5  # initial LR
        self.lr_decay = 0.5  # learning rate decay factor le:=lr/lr_decay
        self.lr_decay_step = 5  # number of batch for decay
        self.weight_decay = 0.0005  # L2 reg
        self.sax = 0.0  # initial absolute translation coeff
        self.saq = args.saq  # initial absolute rotation coeff
        self.srx = 0.0  # initial relative translation coeff
        self.srq = args.srq  # initial relative rotation coeff
        self.learn_gamma = True  # gamma is learnable or fixed
        self.edge_keep_factor = 0.5  # Edge dropout factor during training
        self.gnn_recursion = args.gnn_recursion  # number of recursion for GNN: 2, 3
        # -------------------------------

        # Device - GPU, CPU
        self.device = f'cuda:{args.gpu}' \
            if args.gpu is not None \
            else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Define datasets and dataloaders
        if args.test_scene == 'multi' and args.dataset == '7Scenes':
            self.test_scenes = ['heads', 'chess', 'redkitchen', 'pumpkin', 'office', 'fire', 'stairs']
        elif args.test_scene == 'multi' and args.dataset == 'Cambridge':
            self.test_scenes = ['KingsCollege', 'OldHospital', 'StMarysChurch', 'ShopFacade', 'GreatCourt']
        else:
            self.test_scenes = [args.test_scene]

        sp = 3 if args.dataset == 'Cambridge' else 5
        dataset_loader = CAMBRIDGE_multi if args.dataset == 'Cambridge' else SEVEN_SCENES_multi

        filenames = []
        test_dataset_list = list()
        for s in self.test_scenes:
            test_data_file = str(Path(args.test_data_dir) / f'{s}_fc8_sp{sp}_test'.format(s,sp))
            test_dataset_list.append(dataset_loader(
                root=f'{test_data_file}',
                seqs=[s], train=False, database_set='train', seq_len=self.test_seq_len,
                graph_structure='fc', device_id=args.gpu))
            if args.dataset == '7Scenes' and getattr(args, 'dataset_dir'):
                p_test_split = Path(args.dataset_dir) / s / 'TestSplit.txt'
                with p_test_split.open('r') as f:
                    for line in f.readlines():
                        hit = re.search("[\d]+$", line)
                        if hit is not None:
                            p_seq = Path(args.dataset_dir) / s / f'seq-{int(hit.group()):02d}'
                            rgbs = sorted(p_seq.glob('*.color.*'))
                            if not len(rgbs):
                                rgbs = sorted((p_seq / 'rgb').glob('*.color.*'))
                            filenames.extend(rgbs)

        self.test_dataset_list = test_dataset_list
        self.test_dataset_filenames = filenames
        if args.dataset == '7Scenes' and sum(len(d) for d in self.test_dataset_list) != len(self.test_dataset_filenames):
            msg = f'Not the same number of filenames as test graph files!' \
                  f'{self.test_dataset_list} ({sum(len(d) for d in self.test_dataset_list)})\n' \
                  f'!=\n' \
                  f'{self.test_dataset_filenames} ({len(self.test_dataset_filenames)}'
            logger.error(msg)
            raise IOError(msg)

        x = self.test_dataset_list[0][0].x
        self.num_nodes, self.feat_dim = x.size()
        self.img_W = int(self.feat_dim / (3 * self.img_H))
        self.pose_stat_file = osp.join(args.pose_stat_path, f'{self.dataset}_pose_stats.txt')
        if args.dataset == 'Cambridge' and self.pose_stat_file != None:
            self.pose_m, self.pose_s = np.loadtxt(self.pose_stat_file)  # mean and stdev
        else:
            self.pose_m, self.pose_s = np.array([0,0,0]), np.array([1,1,1]) # mean and stdev


        logger.info(f'Dataset: {self.dataset}')
        logger.info(f'Test scene: {self.test_scene}')
        logger.info(f'Test data dir: {self.test_data_dir}')
        logger.info(f'Test dataset size: {len(self.test_dataset_list[0])}')
        logger.info(f'Images sizes: {self.img_H}, {self.img_W}')
        logger.info(f'Number of nodes in the graph: {self.num_nodes}, {self.graph_structure}')
        logger.info(f'Number of nodes in the graph - test: '
                    f'{self.test_seq_len} {self.test_graph_structure}')
        logger.info(f'Use RP loss: {self.use_VO_loss}')
        logger.info(f'Use RP model: {self.use_VO_model}')
        logger.info(f'srx: {self.srx}')
        logger.info(f'srq: {self.srq}')
        logger.info(f'edge_keep_factor: {self.edge_keep_factor}')
        logger.info(f'gnn_recursion: {self.gnn_recursion}')
        logger.info(f'droprate: {args.droprate}')
        logger.info(f'gpu: {args.gpu}')

        # Define model
        self.feature_extractor = models.resnet34(pretrained=True)
        if self.model_name == 'R1' or self.model_name == 'light_knn':
            self.model = PoseNetX_LIGHT_KNN(
                feature_extractor=self.feature_extractor, droprate=args.droprate, pretrained=True,
                use_gnn=self.graph_structure != 'ind', knn=args.knn).to(self.device)
        elif self.model_name == 'R2':
            self.model = PoseNetX_R2(
                feature_extractor=self.feature_extractor, droprate=args.droprate, pretrained=True,
                use_gnn=self.graph_structure != 'ind', knn=args.knn, gnn_recursion=self.gnn_recursion) \
                .to(self.device)
        elif self.model_name == 'R3':
            self.model = PoseNetX_R2(
                self.feature_extractor, droprate=args.droprate, pretrained=True,
                use_gnn=self.graph_structure != 'ind',
                knn=args.knn, gnn_recursion=self.gnn_recursion, feat_dim=2048, edge_feat_dim=2048,
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

    def eval_RP(self, data_set, weights: Path, ref_node: int = 0, set: str = 'test', scene: str = 'heads'):
        self.model.eval()

        # See https://pytorch.org/docs/stable/notes/randomness.html
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(torch.initial_seed())

        batch_size = 1
        loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,
                            worker_init_fn=seed_worker, generator=g)

        pred_poses = []  # store all predicted poses
        targ_poses = []  # store all target poses
        rel_paths = []

        # loss functions
        t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        q_criterion = quaternion_angular_error

        # inference loop
        for batch_idx, data in tqdm(enumerate(loader), desc=f'{weights.name}',
                                    total=len(loader)):
            idx = batch_idx
            batch_size_ = min(len(data), loader.batch_size)

            # output : 1 x 6 or 1 x STEPS x 6
            output, output_R, edge_index = self.model(data.to(self.device))

            s = output.size()
            output_R = output_R.cpu().data.numpy().reshape((-1, s[-1]))

            target = data.y
            target = target.to('cpu').numpy().reshape((-1, s[-1]))

            if loader.batch_size != 1:
                raise NotImplementedError(f'Larger batch size not implemented!!!')

            edges = edge_index.cpu().data.numpy()

            # Choose one reference absoulte pose and compute the absolute poses in the subgraph
            # using predicted relative poses
            # valid_edges = edges_pruned[1] == 0
            valid_edges = edges[1] == 0

            ref_idx = np.argwhere(valid_edges)[ref_node, 0]
            RP_estimate = output_R[ref_idx, :]
            reference_AP = target[edges[0, ref_idx], :]
            output = reference_AP - RP_estimate
            output = np.expand_dims(output, axis=0)

            # normalize the predicted quaternions
            q = tuple(qexp(p[3:]) for p in output)
            output = np.hstack((output[:, :3], np.asarray(q)))
            q = tuple(qexp(p[3:]) for p in target)
            target = np.hstack((target[:, :3], np.asarray(q)))

            # un-normalize the predicted and target translations
            output[:, :3] = (output[:, :3] * self.pose_s) + self.pose_m
            target[:, :3] = (target[:, :3] * self.pose_s) + self.pose_m

            for j in range(batch_size_):
                idx_ = idx * batch_size + j
                # take the first prediction
                pred_poses.append(output[0])
                targ_poses.append(target[0])
                assert len(pred_poses) == idx_ + 1, f'len(pred_poses): {len(pred_poses)} != {idx_} idx_'
                assert len(targ_poses) == idx_ + 1, f'len(targ_poses): {len(targ_poses)} != {idx_} idx_'

                if self.dataset == '7Scenes':
                    fname = data_set.file_list[idx_]
                    linear_id = int(Path(fname).stem.split('_')[-1])
                    p_rgb = Path(self.test_dataset_filenames[linear_id])
                    p_rel_path = p_rgb.relative_to(self.dataset_dir)
                    rel_paths.append(p_rel_path)

        pred_poses = np.array(pred_poses, copy=False)
        targ_poses = np.array(targ_poses, copy=False)

        # calculate losses
        t_loss = np.asarray([t_criterion(p, t)
                             for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
        q_loss = np.asarray([q_criterion(p, t)
                             for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])

        median_t = np.median(t_loss)
        median_q = np.median(q_loss)
        logger.info(f'[Scene: {scene}, set: {set}, {weights.name}] Error in translation:'
                    f' median {median_t:3.2f} m,'
                    f' mean {np.mean(t_loss):3.2f} m'
                    f'\tError in rotation:'
                    f' median {median_q:3.2f} degrees,'
                    f' mean {np.mean(q_loss):3.2f} degrees')

        if self.dataset == '7Scenes':
            postfix = '_'.join(f'{scene}' for scene in self.test_scenes)
            save_poses(pred_poses=pred_poses, rel_paths=rel_paths,
                       p_output=Path(
                           self.save_dir) / f'{weights.stem.split(".")[0]}_{postfix}__{median_t:0.2f}_{median_q:0.1f}.npz',
                       target_poses=targ_poses
                       )

        return median_t, np.mean(t_loss), median_q, np.mean(q_loss)


def _parse_args(argv):
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset', type=str, help='Name of dataset: 7Scenes, Cambridge', default='Cambridge')
    parser.add_argument('--dataset-dir', type=Path, help='Path to rgb and poses',
                        default='/mnt/disks/data-7scenes/7scenes/')
    parser.add_argument('--test-data-dir', type=str, help='Path to test data',
                        default='/mnt/data-7scenes-ozgur/mozgur/3dv/Cambridge/')
    parser.add_argument('--pose-stat-path', type=str, help='Path to pose statistics file (.txt) for Cambridge, set to "None" for 7Scenes',
                        default='/home/mozgur/relpose-gnn/data/Cambridge/')
    parser.add_argument('--weights', type=Path, help='Weight file name for pre-trained model',
                        default='epoch_149.pth.tar')
    parser.add_argument('--save-dir', type=str, help='Path to output data',
                        default='/mnt/data-7scenes-ozgur/mozgur/3dv')
    parser.add_argument('--test-scene', type=str, help='Which scene to test on', default='GreatCourt')
    parser.add_argument('--num-workers', type=int, help='Number of dataloader workers', default=8)
    parser.add_argument('--model-name', type=str, help='Name of the model (R1, R2, R3)', default='R3')
    parser.add_argument('--gnn-recursion', type=int, help='Number of GNN recursions', default=2)
    parser.add_argument('--srq', type=int, help='Relative rotation loss weight coefficient', default=-3)
    parser.add_argument('--saq', type=int, help='Absolute rotation loss weight coefficient', default=-3)
    parser.add_argument('--knn', default=4, help='knn', type=int)
    parser.add_argument('--droprate', type=float, help='Droprate', default=0.5)
    parser.add_argument('--gpu', default=None, help='GpuId', type=int)
    parser.add_argument('--seed', default=999, help='random seed', type=int)

    args = parser.parse_args(argv)
    if not hasattr(args, 'saq') or args.saq is None:
        setattr(args, 'saq', args.srq)

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

    logdir = Path(args.save_dir) / args.dataset.lower() / args.test_scene
    p_log = str(logdir / f'{Path(args.weights).name}.log')
    print(f'Logging to {p_log}')
    logger.add(p_log)

    model_multi_trainer = MultiModelTrainer(args=args)

    if 1 < len(model_multi_trainer.test_dataset_list):
        msg = 'Note, that seed does not get reset between datasets!'
        logger.warning(msg)
        print(msg)

    checkpoint = torch.load(args.weights)
    model_multi_trainer.model.load_state_dict(checkpoint['model_state_dict'])

    for j in range(len(model_multi_trainer.test_dataset_list)):
        model_multi_trainer.eval_RP(data_set=model_multi_trainer.test_dataset_list[j],
                                    scene=model_multi_trainer.test_scenes[j],
                                    weights=args.weights)


if __name__ == '__main__':
    main(sys.argv[1:])
