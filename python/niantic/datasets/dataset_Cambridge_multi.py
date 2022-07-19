import glob
import itertools
import os.path as osp
import sys
import argparse
import pickle
from pathlib import Path
from loguru import logger
import numpy as np
import torch
import torchvision.transforms as transforms
from torch_geometric.data import Data, Dataset, DataLoader
from external.VLAD.VLADlib.VLAD import query
from niantic.datasets.cambridge_landmark import CambridgeLandmark


#  GreatCourt  KingsCollege  OldHospital  ShopFacade  StMarysChurch
class CAMBRIDGE_multi(Dataset):
    def __init__(self, root,
                 device_id: int,
                 transform=None,
                 pre_transform=None,
                 seqs=None,
                 train=None,
                 seq_len=8,
                 graph_structure='fc',
                 DATA_PATH=None,
                 DATA_PATH_IR=None,
                 fps_sub=1,
                 database_set='train',
                 sampling_period=10,
                 cross_connect=True,
                 retrieval_mode='IR',
                 pathVD = None,
                 treeIndex=None,
                ):
        self.root = root
        self.device_id = device_id
        self.DATA_PATH = DATA_PATH
        self.DATA_PATH_IR = DATA_PATH_IR
        self.seqs = seqs  # number of frames - window size
        self.train = train
        self.seq_len = seq_len
        self.fps_sub = fps_sub
        self.database_set = database_set
        self.query_set = 'train' if train else 'test'
        self.sampling_period = sampling_period
        self.cross_connect = cross_connect
        self.retrieval_mode = retrieval_mode
        self.pathVD = pathVD
        self.treeIndex = treeIndex

        all_file_list = []
        root = str(root)
        s = len(root + "/processed/")
        for file in glob.glob(root + "/processed/data_*"):
            file = file[s:]
            all_file_list.append(file)

        exluded_files = []
        file_list_idxs = []

        for x in all_file_list:
            idx = int(x[5:-3])
            file_list_idxs.append(idx)


        self.file_list = [x for x in all_file_list if x not in exluded_files]

        del all_file_list, exluded_files

        self.file_list.sort()

        # Basic initial graph structures
        # gs_list =  ['ind', 'rnn', 'circ', 'dilated', 'ho', 'fc', 'fc+rand']
        self.graph_structure = graph_structure
        self.bidirectional = True
        self.dilation = 2
        self.hoc = 2
        self.rand_edge_factor = 0.2

        # VLAD
        self.descriptorName = 'ORB'

        super(CAMBRIDGE_multi, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.file_list

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def obtain_KNNs(self, query_img_path, query_img_index, K=8, sampling_period=10, seq='ShopFacade'):
        dist, sorted_indices = query(query_img_path, 200, self.descriptorName,
                                     self.visualDictionary, self.tree)
        sorted_indices = list(itertools.chain.from_iterable(sorted_indices))
        sorted_indices_path = []

        database_idxs = []

        for x in sorted_indices:
            path_comp_list = query_img_path.split('/')[:-2] + self.imageID[x].split('/')[-2:]
            path = '/' + osp.join('', *path_comp_list[1:] )

            sorted_indices_path.append(self.imageID[x])

            if seq != 'ShopFacade':
                if self.cross_connect and (self.imageID[x][-19:-15] == query_img_path[-19:-15]):
                    continue

            if path not in self.dset_database.c_imgs:
                print('skipping frames: ', path)
                continue

            #print(path)

            database_idxs.append(self.dset_database.c_imgs.index(path))
        sorted_indices = np.array(database_idxs)

        if self.database_set == self.query_set:
            sorted_indices = np.delete(sorted_indices, np.where(sorted_indices == query_img_index))

        # Randomly remove half of the frames
        surviving_indices = np.random.random(sorted_indices.shape[0]) < 0.5
        sorted_indices = sorted_indices[surviving_indices]
        # starting_index = sampling_period # This might be beter something else
        starting_index = np.random.randint(0, sampling_period, 1)[0]
        sorted_indices = sorted_indices[starting_index::sampling_period]

        return sorted_indices[:K]

    def process(self):
        file_idx = 0
        self.seq_len = self.seq_len // self.fps_sub

        for seq in self.seqs:
            data_list = list()

            print('Processing sequence: ', seq)
            # Image retreival module
            # load the visual dictionary
            with open(self.pathVD, 'rb') as f:
                self.visualDictionary = pickle.load(f)
            # load the index
            with open(self.treeIndex, 'rb') as f:
                indexStructure = pickle.load(f)
            self.tree = indexStructure[1]
            self.imageID = indexStructure[0]

            transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.RandomCrop(256),
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
            ])

            target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

            dset = CambridgeLandmark(seq, self.DATA_PATH, train=self.train, transform=transform,
                                     target_transform=target_transform)
            print('Loaded Cambridge landmark sequence {:s}, length = {:d}'.format(seq, len(dset)))

            self.dset_database = CambridgeLandmark(seq, self.DATA_PATH,
                                                   train=self.database_set == 'train',
                                                   transform=transform,
                                                   target_transform=target_transform)

            self.num_skipped_frames = 0

            for index in range(0, len(dset)):

                if self.retrieval_mode == 'IR':
                    img_path = dset[index][2]

                    try:
                        kNN_indices = self.obtain_KNNs(query_img_path=img_path, query_img_index=index,
                                                       K=self.seq_len - 1,
                                                       sampling_period=self.sampling_period, seq=seq)
                    except: ValueError

                elif self.retrieval_mode == 'RAND':
                    kNN_indices = np.random.choice(range(len(self.dset_database)),
                                                   self.seq_len - 1, replace=False)

                # sub_graph_indices = [index] + kNN_indices.tolist()
                sub_graph_indices = kNN_indices.tolist()
                print(index, sub_graph_indices)

                if self.retrieval_mode == 'IR' and len(sub_graph_indices) == 0:
                    print('Skipping the frames', index)
                    self.num_skipped_frames += 1
                    continue

                my_subset = torch.utils.data.Subset(self.dset_database, sub_graph_indices)
                my_subset_loader = DataLoader(my_subset, batch_size=len(sub_graph_indices),
                                              num_workers=0, shuffle=self.train)
                my_subset_loader = iter(my_subset_loader)
                batch = next(my_subset_loader)

                batch_itself = dset[index]

                x = torch.cat((batch_itself[0].unsqueeze_(0), batch[0]), 0)
                y = torch.cat((batch_itself[1].unsqueeze_(0), batch[1]), 0)

                if self.fps_sub > 1:
                    x = x[::self.fps_sub]
                    y = y[::self.fps_sub]

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
                elif self.graph_structure == 'fc':
                    source = torch.arange(self.seq_len).view(1, self.seq_len)
                    target = torch.roll(source, -1).view(1, self.seq_len)
                    edge_index = torch.cat((source[:, :-1], target[:, :-1]), 0)
                    for i in range(1, self.seq_len):
                        source_temp = torch.arange(self.seq_len).view(1, self.seq_len)
                        target_temp = torch.roll(source, -i - 1).view(1, self.seq_len)
                        temp = torch.cat((source_temp[:, :-i - 1], target_temp[:, :-i - 1]), 0)
                        edge_index = torch.cat((edge_index, temp), 1)
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
                    for j in range(edge_index.size(1)):
                        source = edge_index[0, j].data
                        target = edge_index[1, j].data
                        y_R[j, :] = y[target, :] - y[source, :]
                else:
                    y_R = None

                data = Data(x=x.view(x.size(0), -1), edge_index=edge_index, y=y, edge_attr=y_R)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(file_idx)))
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
    parser.add_argument('--scene-name', type=str, default='multi',
                        help='Name of the dataset: GreatCourt  KingsCollege  OldHospital  ShopFacade  StMarysChurch')
    parser.add_argument('--mode', type=str, default='train', choices=('train', 'test'),
                        help='train or test')
    parser.add_argument('--seq-len', type=int, default=8,
                        help="Number of nodes in graph. Default: 8")
    parser.add_argument('--sampling-period', '--sp', dest='sampling_period', type=int,
                        help='Strided sampling of neighbors', default=3)
    parser.add_argument('--sampling-method', type=str, default='IR', choices=('IR', 'RAND'),
                        help='How to sample to create graph: either image retrieval or random')
    parser.add_argument('--cross-connect', type=bool, default=False,
                        help='Perform cross connection between different sequences for the training-set')
    parser.add_argument('--graph-data-path', type=Path,
                        #default='/mnt/data-7scenes-ozgur/mozgur/3dv/Cambridge_multi_stat',
                        default='/home/pf/pfstaff/projects/ozgur_poseGraph/data/data/cambridge/cam_multi_all5_IR_sp3_train',
                        help='Where is the train/test graph data will be stored')
    parser.add_argument('--data-path', type=Path,
                        default='/mnt/data-7scenes-ozgur/cambridge',
                        help='Where is the Cambridge data stored')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID. Default: 0.')
    parser.add_argument('--vlad_descriptor_path', type=Path, default=None,
                        help='Where is the VLAD image retrieval descriptors path')
    parser.add_argument( '--vlad_balltree_path', type=Path, default=None,
                        help='Where is the VLAD image retrieval ball-tree path')
    args = parser.parse_args(argv)

    if args.vlad_descriptor_path is None:
        setattr(args, 'vlad_descriptor_path',
                '/home/mozgur/pose_graph_project_danube/VLAD/3dv/visualDictionary/visualDictionary64ORB_{}.pickle'.format(args.scene_name))
    if args.vlad_balltree_path is None:
        setattr(args, 'vlad_balltree_path',
                '/home/mozgur/pose_graph_project_danube/VLAD/3dv/ballTreeIndexes/index_ORB_W64_{}.pickle.pickle'.format(args.scene_name))


    # --------------------Create and check graph data--------------------
    dataset = CAMBRIDGE_multi(
        #args.graph_data_path / f'{args.scene_name}_fc{args.seq_len}_sp{args.sampling_period}_{args.mode}',
        args.graph_data_path,
        seqs=[args.scene_name], DATA_PATH=args.data_path, DATA_PATH_IR=args.data_path,
        train=args.mode=='train', seq_len=args.seq_len, graph_structure='fc', retrieval_mode=args.sampling_method,
        cross_connect=args.cross_connect, sampling_period=args.sampling_period,
        pathVD=args.vlad_descriptor_path, treeIndex=args.vlad_balltree_path)

    logger.info(f'dataset path: {dataset.root}')
    logger.info(f'dataset size: {len(dataset)}')


if __name__ == '__main__':
    main(sys.argv[1:])
