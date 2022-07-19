"""
pytorch data loader for the Cambridge Landmark dataset
"""
import math
import os.path as osp
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data

sys.path.insert(0, '../../../../')
from niantic.utils.pose_utils import process_poses_cambridge, process_poses_cambridge_noRod
from niantic.utils.utils import load_image


class CambridgeLandmark(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None,
                 target_transform=None, seed=7, skip_images=False, noRod=False,
                 normalize_translation=True):
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
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        np.random.seed(seed)

        # directories
        base_dir = osp.join(osp.expanduser(data_path), scene)
        data_dir = osp.join(data_path, scene)

        # decide which sequences to use
        if train:
            split_file = osp.join(base_dir, 'dataset_train.txt')
        else:
            split_file = osp.join(base_dir, 'dataset_test.txt')
        with open(split_file, 'r') as f:
            files_temp = f.readlines()

        files = []
        for x in files_temp:
            if x[:3] == 'seq':
                files.append(x)
        del files_temp

        # print(files)

        dataset_len = len(files)
        print('Dataset length: ', dataset_len)
        # read poses and collect image names
        self.c_imgs = []
        self.gt_idx = np.zeros((dataset_len,), dtype=np.int)

        # ps = np.zeros([dataset_len, 7])
        # self.poses = np.zeros([dataset_len, 6])
        self.poses = np.empty((0, 6))
        for i in range(dataset_len):
            camera = files[i]
            camera = camera.split()

            # p = [float(l) for l in x[1:]]
            # ps[i,:] = np.array(p)

            if noRod:
                cam_pose = [float(r) for r in camera[1:]]
                cam_pose = np.asarray([cam_pose])
                cam_pose = cam_pose[0]
                cam_trans = cam_pose[:3]

            else:
                # quaternion to axis-angle
                cam_rot = [float(r) for r in camera[4:]]
                angle = 2 * math.acos(cam_rot[0])
                x = cam_rot[1] / math.sqrt(1 - cam_rot[0] ** 2)
                y = cam_rot[2] / math.sqrt(1 - cam_rot[0] ** 2)
                z = cam_rot[3] / math.sqrt(1 - cam_rot[0] ** 2)
                cam_rot = [x * angle, y * angle, z * angle]
                cam_rot = np.asarray(cam_rot)
                cam_rot, _ = cv.Rodrigues(cam_rot)

                cam_trans = [float(r) for r in camera[1:4]]
                cam_trans = np.asarray([cam_trans])
                cam_trans = np.transpose(cam_trans)
                cam_trans = - np.matmul(cam_rot, cam_trans)

                cam_pose = np.concatenate((cam_rot, cam_trans), axis=1)
                cam_pose = np.concatenate((cam_pose, [[0, 0, 0, 1]]), axis=0)

            if np.absolute(cam_trans).max() > 10000:
                print("Skipping image: " + i + ". Extremely large translation. Outlier?")
                print(cam_trans)
                continue

            # pose_stats_filename = osp.join('./', 'cambridge_pose_stats_'+ scene +'.txt')
            pose_stats_filename = osp.join('./', 'cambridge_multi_stats.txt')
            #pose_stats_filename = osp.join('./', 'cambridge_pose_stats.txt')

            if noRod:
                cam_pose = process_poses_cambridge_noRod(cam_pose)
            else:
                cam_pose = process_poses_cambridge(cam_pose)

            # self.poses[i,:] = cam_pose
            self.poses = np.vstack((self.poses, cam_pose))
            self.c_imgs.append(osp.join(data_dir, camera[0]))

        # if train:
        #     mean_t = np.mean(self.poses[:,:3],axis=0)
        #     std_t = np.std(self.poses[:,:3],axis=0)
        #     np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        # else:
        #     mean_t, std_t = np.loadtxt(pose_stats_filename)
        mean_t, std_t = np.loadtxt(pose_stats_filename)

        # Normalize translation
        if normalize_translation:
            self.poses[:, :3] -= mean_t
            self.poses[:, :3] /= std_t

        # # # convert pose to translation + log quaternion
        # for i in range(dataset_len):
        #     q = self.poses[i,3:] #shape = (4,)
        #     q *= np.sign(q[0])  # constrain to hemisphere
        #     q = qlog(q) # shape = (3,)
        #     self.poses[i, 3:] = q

    def __getitem__(self, index):
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            img = None
            while img is None:
                img = load_image(self.c_imgs[index])
                pose = self.poses[index]
                index += 1
            index -= 1

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            img = self.transform(img)

        img_path = self.c_imgs[index]

        return img, pose, img_path

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


def main():
    import numpy as np
    """
    visualizes the dataset
    """
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.25, 0.25, 0.25])
    ])
    dataset1 = CambridgeLandmark('ShopFacade', '/mnt/nas3/mozgur/cambridge_landmarks/', True,
                                 transform)
    dataset2 = CambridgeLandmark('StMarysChurch', '/mnt/nas3/mozgur/cambridge_landmarks/', True,
                                 transform)
    dataset3 = CambridgeLandmark('KingsCollege', '/mnt/nas3/mozgur/cambridge_landmarks/', True,
                                 transform)
    dataset4 = CambridgeLandmark('OldHospital', '/mnt/nas3/mozgur/cambridge_landmarks/', True,
                                 transform)

    poses1 = dataset1.poses
    poses2 = dataset2.poses
    poses3 = dataset3.poses
    poses4 = dataset4.poses

    poses = np.concatenate([poses1, poses2, poses3, poses4], 0)
    print(poses.shape)

    # pose_stats_filename = osp.join('./', 'cambridge_multi_stats.txt')
    # mean_t = np.mean(poses[:, :3], axis=0)
    # std_t = np.std(poses[:, :3], axis=0)
    # np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')

    # print(imgs.shape)
    # cue_img = imgs.permute(1, 2, 0).cpu().numpy()
    # cue_img = (cue_img - np.min(cue_img))
    # cue_img = cue_img / np.max(cue_img)
    # plt.imshow(cue_img)
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    main()
