"""
pytorch data loader for the 7-scenes dataset
"""
import os
import os.path as osp
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data

from niantic.utils.pose_utils import process_poses
from niantic.utils.utils import load_image


class SevenScenes(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None,
                 target_transform=None, mode=0, seed=7, real=False,
                 skip_images=False, vo_lib='orbslam'):
        """
        :param scene: scene name ['chess', 'pumpkin', ...]
        :param data_path: root 7scenes data directory.
        Usually '../data/deepslam_data/7Scenes'
        :param train: if True, return the training images. If False, returns the
        testing images
        :param transform: transform to apply to the images
        :param target_transform: transform to apply to the poses
        :param mode: 0: just color image, 1: just depth image, 2: [c_img, d_img]
        :param real: If True, load poses from SLAM/integration of VO
        :param skip_images: If True, skip loading images and return None instead
        :param vo_lib: Library to use for VO (currently only 'dso')
        """
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        np.random.seed(seed)

        # directories
        base_dir = osp.join(osp.expanduser(data_path), scene)
        # data_dir = osp.join('..', 'data', '7Scenes', scene)
        data_dir = osp.join(data_path, scene)

        # decide which sequences to use
        if train:
            split_file = osp.join(base_dir, 'TrainSplit.txt')
        else:
            split_file = osp.join(base_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

        # read poses and collect image names
        self.c_imgs = []
        self.d_imgs = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        ps = {}
        vo_stats = {}
        gt_offset = int(0)
        for seq in seqs:
            seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
            seq_data_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))

            if not os.path.isfile(osp.join(seq_dir, 'frame-{:06d}.color.png'.format(0))):
                seq_dir_pose = osp.join(seq_dir, 'poses')
                seq_dir_rgb = osp.join(seq_dir, 'rgb')
                seq_dir_depth = osp.join(seq_dir, 'depth')
            else:
                seq_dir_pose = seq_dir
                seq_dir_rgb = seq_dir
                seq_dir_depth = seq_dir

            p_filenames = [n for n in os.listdir(seq_dir_pose) if n.find('pose.txt') >= 0]
            # print('Number of poses found: ',len(p_filenames))

            if real:
                pose_file = osp.join(data_dir, '{:s}_poses'.format(vo_lib),
                                     'seq-{:02d}.txt'.format(seq))
                pss = np.loadtxt(pose_file)
                frame_idx = pss[:, 0].astype(np.int)
                if vo_lib == 'libviso2':
                    frame_idx -= 1
                ps[seq] = pss[:, 1:13]
                vo_stats_filename = osp.join(seq_data_dir,
                                             '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'rb') as f:
                    vo_stats[seq] = pickle.load(f)
                # # uncomment to check that PGO does not need aligned VO!
                # vo_stats[seq]['R'] = np.eye(3)
                # vo_stats[seq]['t'] = np.zeros(3)
            else:
                frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
                pss = [
                    np.loadtxt(
                        osp.join(seq_dir_pose, 'frame-{:06d}.pose.txt'.format(i))).flatten()[:12]
                    for i in frame_idx]
                ps[seq] = np.asarray(pss)
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.gt_idx = np.hstack((self.gt_idx, gt_offset + frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir_rgb, 'frame-{:06d}.color.png'.format(i))
                      for i in frame_idx]
            d_imgs = [osp.join(seq_dir_depth, 'frame-{:06d}.depth.png'.format(i))
                      for i in frame_idx]
            self.c_imgs.extend(c_imgs)
            self.d_imgs.extend(d_imgs)

        # pose_stats_filename = osp.join('./pose_stats.txt')
        # if train and not real:
        mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
        std_t = np.ones(3)
        # np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        # else:
        # mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert pose to translation + log quaternion
        self.poses = np.empty((0, 6))
        for seq in seqs:
            pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))

    def __getitem__(self, index):
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            if self.mode == 0:
                img = None
                while img is None:
                    img = load_image(self.c_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
            elif self.mode == 1:
                img = None
                while img is None:
                    img = load_image(self.d_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
            elif self.mode == 2:
                c_img = None
                d_img = None
                while (c_img is None) or (d_img is None):
                    c_img = load_image(self.c_imgs[index])
                    d_img = load_image(self.d_imgs[index])
                    pose = self.poses[index]
                    index += 1
                img = [c_img, d_img]
                index -= 1
            else:
                raise Exception('Wrong mode {:d}'.format(self.mode))

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            if self.mode == 2:
                img = [self.transform(i) for i in img]
            else:
                img = self.transform(img)

        relative_img_path = Path(self.c_imgs[index]).relative_to(self.data_path)
        return img, pose, str(relative_img_path)

    def __len__(self):
        return self.poses.shape[0]


def visulize(data_set, pred_poses, targ_poses, dataset='7Scenes', freq=1000):
    fig = plt.figure()
    if dataset != '7Scenes':
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    # plot on the figure object
    ss = max(1, int(len(data_set) / freq))  # 100 for stairs
    # scatter the points and draw connecting line
    x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
    y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
    if dataset != '7Scenes':  # 2D drawing
        # ax.plot(x, y, c='b')
        ax.scatter(x[0, :], y[0, :], c='r')
        ax.scatter(x[1, :], y[1, :], c='g')
    else:
        z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
        for xx, yy, zz in zip(x.T, y.T, z.T):
            ax.plot(xx, yy, zs=zz, c='b')
        ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0)
        ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0)
        ax.view_init(azim=119, elev=13)

    plt.show(block=True)

# def main():
#     """
#     visualizes the dataset
#     """
#     from vis_utils import show_batch, show_stereo_batch
#     from torchvision.utils import make_grid
#     import torchvision.transforms as transforms
#     seq = 'chess'
#     mode = 0
#     num_workers = 6
#     transform = transforms.Compose([
#         transforms.Scale(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#     dset1 = SevenScenes('heads', '/mnt/nas2/shared/datasets/7scenes', False, transform,mode=mode)
#     dset2 = SevenScenes('pumpkin', '/mnt/nas2/shared/datasets/7scenes', False, transform,mode=mode)
#
#     poses1 = np.zeros([1000,6])
#     poses2 = np.zeros([1000,6])
#
#     for i in range(1000):
#         poses1[i,:] = dset1[i][1]
#         poses2[i,:] = dset2[i][1]


# visulize(dset1, poses1, poses2, dataset='7Scenes')
# if __name__ == '__main__':
#    main()
