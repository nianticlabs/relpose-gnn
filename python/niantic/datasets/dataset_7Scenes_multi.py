import argparse
import glob
import os
import os.path as osp
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from torch.utils import data as torch_data
from torch_geometric.data import Data, Dataset, DataLoader
from tqdm import tqdm

from external.sanet_relocal_demo.reloc_pipeline.util_func import \
    preprocess_scene, x_2d_coords_torch, preprocess_query
from external.sanet_relocal_demo.relocal.vlad_encoder import VLADEncoder
from external.sanet_relocal_demo.relocal_data.seven_scene.seven_scene_manual_dataset import \
    SevenSceneManualDataset
from niantic.datasets.dataset_arparse import add_arguments_dataset
from niantic.datasets.graph_structure import GraphStructure
from niantic.datasets.seven_scenes import SevenScenes
from path_config import PATH_PROJECT


# ['heads','chess','redkitchen','pumpkin','office', 'fire','stairs']
class SEVEN_SCENES_multi(Dataset):
    def __init__(self, root,
                 device_id: int,
                 transform=None, pre_transform=None,
                 seqs=('heads',), train=True, seq_len=8, graph_structure='fc',
                 DATA_PATH=None,
                 DATA_PATH_IR=None,
                 GT_PATH= './GT/7Scenes',
                 fps_sub=1,
                 database_set='train',
                 sampling_period=5,
                 excluded_scenes=None,
                 cross_connect=False,
                 retrieval_mode='IR',
                 num_workers: int = 8
                 ):

        logger.info(f'graph data path: {root}')
        self.root = root
        self.device_id = device_id
        self.device = torch.cuda.device(device_id)
        self.DATA_PATH = Path(DATA_PATH) if DATA_PATH is not None else None
        self.DATA_PATH_IR = Path(DATA_PATH_IR) if DATA_PATH_IR is not None else None
        self.GT_PATH = GT_PATH
        self.mode = 0  # only RGB images
        self._num_workers = num_workers
        self.seqs = seqs  # number of frames - window size
        self.train = train
        self.seq_len = seq_len
        self.fps_sub = fps_sub
        self.database_set = database_set
        self.query_set = 'train' if train else 'test'
        self.sampling_period = sampling_period
        self.cross_connect = cross_connect
        self.retrieval_mode = retrieval_mode
        if excluded_scenes is not None and not isinstance(excluded_scenes, (list, tuple)):
            excluded_scenes = (excluded_scenes,)

        all_file_list = []
        root = str(root)
        s = len(root + "/processed/")
        #for file in (root / "processed").glob("data_*"):
        for file in glob.glob(root + "/processed/data_*"):
            file = file[s:]
            all_file_list.append(file)

        excluded_files = []
        file_list_idxs = []

        if excluded_scenes != None:
            for excluded_scene in excluded_scenes:
                if excluded_scene == 'heads':
                    scene_file_idx_min = -1
                    scene_file_idx_max = 1000
                elif excluded_scene == 'chess':
                    scene_file_idx_min = 999
                    scene_file_idx_max = 5000
                elif excluded_scene == 'redkitchen':
                    scene_file_idx_min = 4999
                    scene_file_idx_max = 12000
                elif excluded_scene == 'pumpkin':
                    scene_file_idx_min = 11999
                    scene_file_idx_max = 16000
                elif excluded_scene == 'office':
                    scene_file_idx_min = 15999
                    scene_file_idx_max = 22000
                elif excluded_scene == 'fire':
                    scene_file_idx_min = 21999
                    scene_file_idx_max = 24000
                elif excluded_scene == 'stairs':
                    scene_file_idx_min = 23999
                    scene_file_idx_max = 26000

                for path_file in all_file_list:
                    idx = int(path_file[5:-3])
                    file_list_idxs.append(idx)
                    if scene_file_idx_min < idx < scene_file_idx_max:
                        excluded_files.append(path_file)

            self.file_list = [x for x in all_file_list if x not in excluded_files]
        else:
            self.file_list = all_file_list.copy()

        del all_file_list, excluded_files

        self.file_list.sort()
        #print(self.file_list)

        # Basic initial graph structures
        # gs_list =  ['ind', 'rnn', 'circ', 'dilated', 'ho', 'fc', 'fc+rand']
        self.graph_structure = graph_structure
        self.bidirectional = True

        self.dilation = 2
        self.hoc = 2
        self.rand_edge_factor = 0.2
        # should be set in python/__init__.py
        self.netvlad_checkpoint = \
            Path(os.environ.get('TORCH_HOME', PATH_PROJECT / 'models')) / 'netvlad_vgg16.tar'
        if not (self.netvlad_checkpoint.is_file() or self.netvlad_checkpoint.is_symlink()):
            cmd = (
            'wget', 'https://storage.googleapis.com/niantic-lon-static/research/relpose-gnn/models/netvlad_vgg16.tar',
            '-O', self.netvlad_checkpoint)
            self.netvlad_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(cmd)

        super(SEVEN_SCENES_multi, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.file_list

    def download(self):
        # Download to `self.raw_dir`.
        pass

    @property
    def num_workers(self):
        return self._num_workers

    def obtain_NetVLAD_feats(self, seq, sampling_period=1, split='train', num_workers: int = 8):
        # read db frames from binary file
        frames_path = Path(self.DATA_PATH_IR) / seq / f'{split}_frames.bin'
        with open(frames_path, 'rb') as f:
            frames = pickle.load(f, encoding='latin1')[::sampling_period]
        logger.info(f'Total frames for image retrieval database of {seq}: {len(frames):d}')

        transform_func = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                              std=(0.229, 0.224, 0.225))

        # load dataset
        scene_set = SevenSceneManualDataset(base_dir=self.DATA_PATH_IR,
                                            seq_frame_list=frames,
                                            transform=transform_func,
                                            fill_depth_holes=False,
                                            output_dim=(3, 192, 256))
        scene_loader = DataLoader(scene_set, batch_size=1, num_workers=num_workers, shuffle=False)

        # add train frames to vlad database
        frame_dim = (192, 256)
        x_2d_scene = x_2d_coords_torch(5, frame_dim[0], frame_dim[1]).cuda()
        x_2d_scene = x_2d_scene.view(5, -1, 2)

        retrival_scene_feats = []

        for sample_dict in tqdm(scene_loader):
            with self.device, torch.no_grad():
                scene_rgb = sample_dict['frames_img'].cuda()
                scene_depth = sample_dict['frames_depth'].cuda()
                scene_K = sample_dict['frames_K'].cuda()
                scene_Tcw = sample_dict['frames_Tcw']
                scene_ori_rgb = sample_dict['frames_ori_img'].cuda()
                scene_input, scene_ori_rgb, X_world, scene_valid_mask, \
                self.scene_center, self.rand_R = preprocess_scene(
                    x_2d_scene, scene_rgb, scene_depth, scene_K, scene_Tcw, scene_ori_rgb)

                cur_scene_input = scene_input[0, :, :3, ...]
                retrival_scene_feat = self.vlad_db.forward(cur_scene_input)
                retrival_scene_feats += [feat.unsqueeze(0).cpu().numpy()
                                         for feat in retrival_scene_feat]

        return frames, retrival_scene_feats

    def obtain_KNNs(self, query_img_index, K=4, sampling_period=10, scene_seq_len=10, seq='chess',
                    num_workers: int = 8):
        """
        Called by `process()`
        """
        q_frame = self.query_frames[query_img_index]

        transform_func = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                              std=(0.229, 0.224, 0.225))
        q_set = SevenSceneManualDataset(base_dir=self.DATA_PATH_IR,
                                        seq_frame_list=[q_frame],
                                        transform=transform_func,
                                        fill_depth_holes=False,
                                        output_dim=(3, 192, 256),
                                        nsample_per_group=1)
        q_loader = DataLoader(q_set, batch_size=1, num_workers=num_workers, shuffle=False)

        with self.device, torch.no_grad():
            sample_dict = next(iter(q_loader))
            query_img = sample_dict['frames_img']
            query_ori_img = sample_dict['frames_ori_img']
            query_depth = sample_dict['frames_depth'].cuda()
            query_Tcw = sample_dict['frames_Tcw']
            ori_query_K = sample_dict['frames_K'].clone().cuda()

            query_img = query_img.squeeze(1)
            query_ori_img = query_ori_img.squeeze(1)
            query_depth = query_depth.squeeze(1)
            query_Tcw = query_Tcw.squeeze(1)
            ori_query_K = ori_query_K.squeeze(1)

            query_img, query_X_world, valid_mask, query_ori_img, \
            scene_center, query_Tcw, query_K, rand_R = preprocess_query(query_img,
                                                                        query_depth,
                                                                        query_ori_img,
                                                                        query_Tcw, ori_query_K,
                                                                        self.scene_center,
                                                                        self.rand_R, (48, 64))
            seq_feat = self.vlad_db.forward(query_img)

        # determine top-K
        n_scenes = len(self.database_feats)
        dist = [cos_sim(seq_feat.cpu().numpy(), self.database_feats[idx])
                for idx in range(n_scenes)]
        dist = np.asarray(dist).ravel()
        sorted_indices = np.argsort((1 - dist))

        if (self.database_set == self.query_set) and self.cross_connect:  # True for training-set
            # Choose frames only from other sequences
            if seq != 'heads':
                valid_indices = \
                    (sorted_indices // scene_seq_len) != (query_img_index // scene_seq_len)
                sorted_indices = sorted_indices[valid_indices]

        elif self.database_set == self.query_set:
            sorted_indices = np.delete(sorted_indices, np.where(sorted_indices == query_img_index))

        # Randomly remove half of the frames
        surviving_indices = np.random.random(sorted_indices.shape[0]) < 0.5
        sorted_indices = sorted_indices[ surviving_indices ]
        #starting_index = sampling_period # This might be beter something else
        starting_index = np.random.randint(0, sampling_period, 1)[0]
        sorted_indices = sorted_indices[starting_index::sampling_period]

        # top_K_frames = [self.database_frames[idx] for idx in sorted_indices[:K]]

        return sorted_indices[:K], 0

    def process(self):
        file_idx = 0
        # ToDo(?): check, if seq_len needs to be overwritten here, or could be a local var
        self.seq_len = self.seq_len // self.fps_sub
        self.vlad_db = VLADEncoder(checkpoint_path=self.netvlad_checkpoint,
                                   dev_id=self.device_id)

        for seq in self.seqs:
            data_list = list()

            logger.info(f'Processing sequence: {seq}')
            # Image retreival module
            self.database_frames, self.database_feats = \
                self.obtain_NetVLAD_feats(seq=seq, split=self.database_set,
                                          num_workers=self.num_workers)

            # read test frames from binary file
            query_frames_path = os.path.join(self.DATA_PATH_IR,
                                             '%s/' % seq + self.query_set + '_frames.bin')

            with open(query_frames_path, 'rb') as f:
                self.query_frames = pickle.load(f, encoding='latin1')

            # print(self.query_frames)
            stats_file = osp.join(self.GT_PATH, seq, 'stats.txt')
            stats = np.loadtxt(stats_file)

            transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))
            ])
            target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

            dset = SevenScenes(seq, self.DATA_PATH, train=self.train, transform=transform,
                               target_transform=target_transform, mode=self.mode)
            print('Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq, len(dset)))

            dset_database = SevenScenes(seq, self.DATA_PATH, train=self.database_set == 'train',
                                        transform=transform,
                                        target_transform=target_transform, mode=self.mode)

            for index in range(len(dset)):
                # if index == 2:
                #    break

                if self.retrieval_mode == 'IR':
                    kNN_indices, _ = self.obtain_KNNs(query_img_index=index, K=self.seq_len - 1,
                                                      sampling_period=self.sampling_period,
                                                      seq=seq, num_workers=self.num_workers)

                elif self.retrieval_mode == 'RAND':
                    kNN_indices = np.random.choice(range(len(dset)), self.seq_len - 1,
                                                   replace=False)

                # sub_graph_indices = [index] + kNN_indices.tolist()
                sub_graph_indices = kNN_indices.tolist()
                # print(index, sub_graph_indices)
                path_list = [dset.c_imgs[index]]
                for x in sub_graph_indices:
                    path_list.append(dset_database.c_imgs[x])

                # Print path of the images in the graph
                # print(path_list)
                print(index, ': ' ,sub_graph_indices)

                my_subset = torch.utils.data.Subset(dset_database, sub_graph_indices)
                my_subset_loader = DataLoader(my_subset, batch_size=len(sub_graph_indices),
                                              num_workers=self.num_workers, shuffle=self.train)
                my_subset_loader = iter(my_subset_loader)
                batch = next(my_subset_loader)

                batch_itself = dset[index]

                # x.shape: [self.seq_len, 3, 256, -1] e.g. [8, 3, 256, 341]
                x = torch.cat((batch_itself[0].unsqueeze_(0), batch[0]), 0)
                # y.shape: [self.seq_len, 6] e.g. [8, 6], where 6 is [t, exp(q)]
                y = torch.cat((batch_itself[1].unsqueeze_(0), batch[1]), 0)

                if self.fps_sub > 1:
                    x = x[::self.fps_sub]
                    y = y[::self.fps_sub]

                # if x.size(0) != self.seq_len:
                #    continue

                # Define edge connectivity
                if self.graph_structure == 'ind':
                    edge_index = None
                elif self.graph_structure == 'rnn':
                    source = torch.arange(self.seq_len).view(1, self.seq_len)
                    target = torch.roll(source, -1).view(1, self.seq_len)
                    edge_index = torch.cat((source[:, :-1], target[:, :-1]), 0)
                elif self.graph_structure == 'circ':
                    source = torch.arange(self.seq_len).view(1, self.seq_len)
                    target = torch.roll(source, -1).view(1, self.seq_len)
                    edge_index = torch.cat((source[:, :], target[:, :]), 0)
                elif self.graph_structure == 'dilated':
                    source = torch.arange(self.seq_len).view(1, self.seq_len)
                    target = torch.roll(source, -self.dilation).view(1, self.seq_len)
                    edge_index = torch.cat((source[:, :], target[:, :]), 0)
                elif self.graph_structure == 'ho':
                    source = torch.arange(self.seq_len).view(1, self.seq_len)
                    target = torch.roll(source, -1).view(1, self.seq_len)
                    edge_index = torch.cat((source[:, :-1], target[:, :-1]), 0)
                    for i in range(1, self.hoc):
                        source_temp = torch.arange(self.seq_len).view(1, self.seq_len)
                        target_temp = torch.roll(source, -i - 1).view(1, self.seq_len)
                        temp = torch.cat((source_temp[:, :-i - 1], target_temp[:, :-i - 1]), 0)
                        edge_index = torch.cat((edge_index, temp), 1)
                elif self.graph_structure == GraphStructure.FC:
                    source = torch.arange(self.seq_len).view(1, self.seq_len)
                    target = torch.roll(source, -1).view(1, self.seq_len)
                    edge_index = torch.cat((source[:, :-1], target[:, :-1]), 0)
                    for i in range(1, self.seq_len):
                        source_temp = torch.arange(self.seq_len).view(1, self.seq_len)
                        target_temp = torch.roll(source, -i - 1).view(1, self.seq_len)
                        temp = torch.cat((source_temp[:, :-i - 1], target_temp[:, :-i - 1]), 0)
                        edge_index = torch.cat((edge_index, temp), 1)
                    # edge_index2 = torch.tensor(tuple(itertools.combinations(np.arange(self.seq_len), 2))).T
                    # assert all(a == b for a, b in zip(edge_index2.shape, edge_index.shape))
                    # for i in range(edge_index.shape[1]):
                    #     print(i)
                    #     assert edge_index[:, i:i+1] in edge_index2
                    #     assert edge_index2[:, i:i + 1] in edge_index
                    # assert torch.allclose(edge_index, edge_index2), f'{edge_index}\n{edge_index2}'
                elif self.graph_structure == 'fc+rand':
                    source = torch.arange(self.seq_len).view(1, self.seq_len)
                    target = torch.roll(source, -1).view(1, self.seq_len)
                    edge_index = torch.cat((source[:, :-1], target[:, :-1]), 0)
                    for i in range(1, self.hoc):
                        source_temp = torch.arange(self.seq_len).view(1, self.seq_len)
                        target_temp = torch.roll(source, -i - 1).view(1, self.seq_len)
                        temp = torch.cat((source_temp[:, :-i - 1], target_temp[:, :-i - 1]), 0)
                        edge_index = torch.cat((edge_index, temp), 1)
                    # Random edge assignment
                    for i in range(self.hoc, self.seq_len):
                        source_temp = torch.arange(self.seq_len).view(1, self.seq_len)
                        target_temp = torch.roll(source, -i - 1).view(1, self.seq_len)

                        source_temp = source_temp[:, :-i - 1]
                        target_temp = target_temp[:, :-i - 1]

                        surviving_edges = torch.rand(source_temp.size(1)) < self.rand_edge_factor
                        source_temp = source_temp[:, surviving_edges]
                        target_temp = target_temp[:, surviving_edges]

                        temp = torch.cat((source_temp, target_temp), 0)
                        edge_index = torch.cat((edge_index, temp), 1)

                # bi-directional graph
                if edge_index is not None:
                    if self.bidirectional or self.graph_structure == 'fc' \
                      or self.graph_structure == 'fc+rand':
                        temp = torch.flip(edge_index, [0])
                        edge_index = torch.cat((edge_index, temp), 1)

                    # Assign relative poses between nodes
                    y_R = torch.zeros([edge_index.size(1), y.size(1)])
                    for i in range(edge_index.size(1)):
                        source = edge_index[0, i].data
                        target = edge_index[1, i].data
                        y_R[i, :] = y[target, :] - y[source, :]
                else:
                    y_R = None

                # x.shape: [8, 3, 256, 341]
                # x.view(x.size(0), -1).shape: [8, 261888]
                # y.shape: [8, 6]
                # y_R.shape: [56, 6]
                data = Data(x=x.view(x.size(0), -1), edge_index=edge_index, y=y, edge_attr=y_R)
                # data_list.append(subgraph_data)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f'data_{file_idx:06d}.pt'))
                file_idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        file = self.file_list[idx]
        data = torch.load(osp.join(self.processed_dir, file))
        return data


def main(argv):
    # --------------------Create and check graph data--------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('scene_name', type=str, default='heads',
                        help='Name of the dataset')
    parser.add_argument('mode', type=str, choices=('train', 'test'),
                        help='train or test')
    parser = add_arguments_dataset(parser)
    # parser.add_argument('--seq-len', type=int, default=8,
    #                     help="Number of nodes in graph. Default: 8")
    # parser.add_argument('--sampling-period', '--sp', dest='sampling_period', type=int,
    #                     help='Strided sampling of neighbors', default=5)
    # parser.add_argument('--sampling-method', type=str, default='IR', choices=('IR', 'RAND'),
    #                     help='How to sample to create graph: either image retrieval or random')
    # parser.add_argument('--cross-connect', type=bool, default=False,
    #                     help='Perform cross connection between different sequences for the training-set')
    parser.add_argument('--graph-data-path', type=Path,
                        default='/home/pf/pfstaff/projects/ozgur_poseGraph/graph_data/7Scenes',
                        help='Where is the train/test graph data will be stored')
    parser.add_argument('--data-path', type=Path,
                        default='/home/pf/pfstaff/projects/ozgur_poseGraph/datasets/7Scenes',
                        help='Where is the 7Scenes data stored')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID. Default: 0.')
    parser.add_argument(
        '--num-workers', type=int, help='Number of dataloader workers', default=8)
    args = parser.parse_args(argv)

    if args.scene_name == 'multi':
        seq = ('heads', 'chess', 'redkitchen', 'pumpkin', 'office', 'fire', 'stairs')
    else:
        seq = (args.scene_name, )

    if args.mode == 'train':
        dataset = SEVEN_SCENES_multi(
            args.graph_data_path / f'{args.scene_name}_fc{args.seq_len}_sp{args.sampling_period}_train',
            GT_PATH=PATH_PROJECT / 'GT' / '7Scenes',
            DATA_PATH=args.data_path,
            DATA_PATH_IR=args.data_path,
            device_id=args.device,
            seqs=seq, train=True,
            retrieval_mode=args.sampling_method,
            cross_connect = args.cross_connect,
            database_set='train', seq_len=args.seq_len, graph_structure=GraphStructure.FC,
            sampling_period=args.sampling_period, excluded_scenes=None,
            num_workers=args.num_workers)
    else:
        dataset = SEVEN_SCENES_multi(
            args.graph_data_path / f'{args.scene_name}_fc{args.seq_len}_sp{args.sampling_period}_test',
            GT_PATH=PATH_PROJECT / 'GT' / '7Scenes',
            DATA_PATH = args.data_path,
            DATA_PATH_IR=args.data_path,
            device_id=args.device,
            seqs=seq, train=False,
            retrieval_mode=args.sampling_method,
            cross_connect=args.cross_connect,
            database_set='train', seq_len=args.seq_len, graph_structure=GraphStructure.FC,
            sampling_period=args.sampling_period, excluded_scenes=None,
            num_workers=args.num_workers)


    print('dataset size: ', len(dataset))
    logger.info('dataset size: ', len(dataset))

    # # -------------------------Visualize the graph--------------------------
    #from torch_geometric.utils.convert import to_networkx

    #data = dataset[0]
    #graph = to_networkx(data)
    #node_labels = data.y[list(graph.nodes)].numpy()

    # plt.figure(1, figsize=(14, 12))
    # nx.draw(graph, cmap=plt.get_cmap('Set1'), node_size=75, linewidths=6)
    # plt.show()
    #
    ##---------------------- Visualize images from subgraphs----------------------
    # for batch_idx, data in enumerate(tqdm(loader)):
    #     with torch.cuda.device(args.device):
    #         with torch.no_grad():
    #             imgs = data.x
    #             imgs = imgs.view(imgs.size(0), 3, 256,-1).contiguous()
    #             fig = plt.figure(figsize=(50, 10))
    #             for j in range(1,imgs.shape[0]+1):
    #                 cue_img = imgs[j-1].permute(1, 2, 0).cpu().numpy()
    #                 cue_img = (cue_img - np.min(cue_img))
    #                 cue_img = cue_img / np.max(cue_img)
    #                 fig.add_subplot(1, imgs.shape[0]+1, j)
    #                 plt.imshow(cue_img)
    #             plt.show()
    #             plt.close()
    #
    #             if batch_idx > 2:
    #                 break
    #
    # data = dataset[1]
    # imgs = data.x
    # imgs = imgs.view(imgs.size(0), 3, 256, -1).contiguous()
    # cue_img = imgs[0].permute(1, 2, 0).cpu().numpy()
    # cue_img = (cue_img - np.min(cue_img))
    # cue_img = cue_img / np.max(cue_img)
    # plt.imshow(cue_img)
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    main(sys.argv[1:])
