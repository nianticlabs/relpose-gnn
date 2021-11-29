import argparse
import glob
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from loguru import logger

p_sanet_relocal_demo = str(Path(__file__).parent.parent.parent)
if p_sanet_relocal_demo not in sys.path:
    sys.path.insert(0, p_sanet_relocal_demo)
# sys.path.append('../../')
# sys.path.append('./')

import core_3dv.camera_operator as cam_opt
from seq_data.frame_seq_data import FrameSeqData
from seq_data.tum_rgbd.tum_seq2ares import export_to_tum_format, \
    export_tum_img_info


def re_organize(seq_path):
    rgb_dir = os.path.join(seq_path, 'rgb')
    if not os.path.exists(rgb_dir):
        os.mkdir(rgb_dir)

    depth_dir = os.path.join(seq_path, 'depth')
    if not os.path.exists(depth_dir):
        os.mkdir(depth_dir)

    poses_dir = os.path.join(seq_path, 'poses')
    if not os.path.exists(poses_dir):
        os.mkdir(poses_dir)

    seqs = sorted(glob.glob(os.path.join(seq_path, '*.pose.txt')))
    seq_names = [seq.split('/')[-1].split('.pose.txt')[0] for seq in seqs]
    for seq_name in seq_names:
        shutil.move(os.path.join(seq_path, '%s.color.png' % seq_name),
                    os.path.join(seq_path, 'rgb', '%s.color.png' % seq_name))
        shutil.move(os.path.join(seq_path, '%s.depth.png' % seq_name),
                    os.path.join(seq_path, 'depth', '%s.depth.png' % seq_name))
        shutil.move(os.path.join(seq_path, '%s.pose.txt' % seq_name),
                    os.path.join(seq_path, 'poses', '%s.pose.txt' % seq_name))
    os.system('rm %s' % os.path.join(seq_path, 'Thumbs.db'))


def scenes2ares(seq_path: Path, seq_name: str):
    path_seq = seq_path / seq_name

    rgb_dir = path_seq / 'rgb'
    depth_dir = path_seq / 'depth'
    poses_dir = path_seq / 'poses'

    # Aron's case:
    if not rgb_dir.is_dir():
        rgb_dir = path_seq
    if not depth_dir.is_dir():
        depth_dir = path_seq
    if not poses_dir.is_dir():
        poses_dir = path_seq

    # frames = sorted(glob.glob(os.path.join(rgb_dir, '*.color.png')))
    frames = sorted(rgb_dir.glob('*.color.png'),
                    key=lambda p: int(p.stem.split('.')[0].split('-')[1]))
    frame_names = tuple(seq.name.split('.')[0] for seq in frames)

    default_intrinsic = np.asarray([585., 585., 320., 240., 0, 0], dtype=np.float32)

    frame_seq_data = FrameSeqData()

    # Read the pose
    for frame_idx, frame_name in enumerate(frame_names):
        pose_file = poses_dir / f'{frame_name}.pose.txt'  # os.path.join(poses_dir, frame_name + '.pose.txt')
        # rgb_file = os.path.join(poses_dir, frame_name + '.color.png')
        # depth_file = os.path.join(poses_dir, frame_name + '.depth.png')

        # Read the pose
        pose = np.loadtxt(pose_file).astype(np.float32).reshape(4, 4)
        Tcw = cam_opt.camera_pose_inv(pose[:3, :3], pose[:3, 3])
        timestamp = float(frame_name.split('-')[1])

        img_file_name = (rgb_dir / f'{frame_name}.color.png').relative_to(path_seq.parent.parent)
        img_file_name_abs = seq_path / img_file_name
        if not (img_file_name_abs.is_file() or img_file_name_abs.is_symlink()):
            logger.warning(f'\nCould not find rgb: {img_file_name_abs}')

        depth_file_name = (depth_dir / f'{frame_name}.depth.png').relative_to(
            path_seq.parent.parent)
        depth_file_name_abs = seq_path / depth_file_name
        if not (depth_file_name_abs.is_file() or depth_file_name_abs.is_symlink()):
            logger.warning(f'\nCould not find depth: {depth_file_name_abs}')

        frame_seq_data.append_frame(
            frame_idx=frame_idx,
            img_file_name=str(img_file_name),
            # os.path.join(seq_name, 'rgb', f'{frame_name}.color.png'),
            Tcw=Tcw,
            camera_intrinsic=default_intrinsic,
            frame_dim=(480, 640),
            time_stamp=timestamp,
            depth_file_name=str(
                depth_file_name))  # os.path.join(seq_name, 'depth', f'{frame_name}.depth.png'))

    return frame_seq_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Scenes2Seq 7Scenes')
    parser.add_argument('seq_dir', type=Path, help='Problem directory')
    parser.add_argument('--dst-dir', dest='dst_dir', type=Path,
                        help='Output directory for generated results')
    args = parser.parse_args()

    if not os.path.isdir(args.seq_dir):
        raise OSError('Folder does not exist: {}'.format(args.seq_dir))

    # Output path same as input path, if not provided
    if not hasattr(args, 'dst_dir') or args.dst_dir is None:
        args.dst_dir = args.seq_dir

    seq_path = Path(args.seq_dir)
    # seq_parent_dir = seq_path.parent
    seq_name = seq_path.name
    seq_subs = seq_path.glob('seq*')

    # re-organize sequences
    for seq_path in seq_subs:
        if not os.path.isdir(seq_path):
            continue
        seq_num = seq_path.name

        # Try naively first
        seq = scenes2ares(seq_path.parent.parent, Path(seq_name) / seq_num)
        # Check, if re-organization is needed
        if len(seq) == 0:
            re_organize(seq_path)
            seq = scenes2ares(seq_path.parent.parent, Path(seq_name) / seq_num)
            # check, if re-organization helped
            if len(seq) == 0:
                logger.error(f'Could not read dataset from {args.seq_dir}')
                continue

        # Compute destination path
        p_json = args.dst_dir / seq_num / 'seq.json'
        # Make sure desination folder exists
        if not p_json.parent.is_dir():
            os.makedirs(p_json.parent)
        # Save sequence info
        seq.dump_to_json(p_json)

        # Prepare output paths (parent already created above)
        p_rgb_txt = args.dst_dir / seq_num / 'rgb.txt'
        p_depth_txt = args.dst_dir / seq_num / 'depth.txt'
        # Save output
        export_tum_img_info(seq, str(p_rgb_txt), str(p_depth_txt))
        # Check, if successful
        if p_rgb_txt.is_file():
            logger.info(f'Wrote to {p_rgb_txt}')
        if p_depth_txt.is_file():
            logger.info(f'Wrote to {p_depth_txt}')

        # Prepare output path
        p_groundtruth_txt = args.dst_dir / seq_num / 'groundtruth.txt'
        export_to_tum_format(seq, str(p_groundtruth_txt))
        # Check, if successful
        if p_groundtruth_txt.is_file():
            logger.info(f'Wrote to {p_groundtruth_txt}')

        # Read back intrinsics
        K = seq.get_K_mat(seq.frames[0])
        # Save intrinsics explicitly
        p_K_txt = args.dst_dir / seq_num / 'K.txt'
        np.savetxt(str(p_K_txt), K)
        if p_K_txt.is_file():
            logger.info(f'Wrote to {p_K_txt}')

    # generate train and test frames information (e.g. extrinsic, intrinsic)

    # load train and test split txt
    with open(args.seq_dir / 'TestSplit.txt') as f:
        # test_seqs_l = f.readlines()
        test_seqs_l = tuple(int(l.split('sequence')[1].strip()) for l in f)

    with open(args.seq_dir / 'TrainSplit.txt') as f:
        # train_seqs_l = f.readlines()
        train_seqs_l = tuple(int(l.split('sequence')[1].strip()) for l in f)

    # collect all test and train frames from all sequences
    test_frames = []
    for test_seq in test_seqs_l:
        json_path = args.dst_dir / f'seq-{test_seq:02d}' / 'seq.json'
        seq = FrameSeqData(json_path)
        test_frames += seq.frames
    del test_seq, json_path

    train_frames = []
    for train_seq in train_seqs_l:
        json_path = args.dst_dir / f'seq-{train_seq:02d}' / 'seq.json'
        seq = FrameSeqData(json_path)
        train_frames += seq.frames

    # dump
    p_test_frames_bin = args.dst_dir / 'test_frames.bin'
    with open(p_test_frames_bin, 'wb') as f:
        pickle.dump(test_frames, f)
    if p_test_frames_bin.is_file():
        logger.info(f'Wrote to {p_test_frames_bin}')

    p_train_frames_bin = args.dst_dir / 'train_frames.bin'
    with open(p_train_frames_bin, 'wb') as f:
        pickle.dump(train_frames, f)
    if p_train_frames_bin.is_file():
        logger.info(f'Wrote to {p_train_frames_bin}')
