import tensorflow as tf
import numpy as np
from scipy.ndimage import imread
from glob import glob
import os
import json
import re
from matplotlib import pylab as plt

import constants as c
from tfutils import log10

##
# Data
##

def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames /= (255 / 2)
    new_frames -= 1

    return new_frames

def denormalize_frames(frames):
    """
    Performs the inverse operation of normalize_frames.

    @param frames: A numpy array. The frames to be converted.

    @return: The denormalized frames.
    """
    new_frames = frames + 1
    new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    new_frames = new_frames.astype(np.uint8)

    return new_frames

def clip_l2_diff(clip):
    """
    @param clip: A numpy array of shape [c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    @return: The sum of l2 differences between the frame pixels of each sequential pair of frames.
    """
    diff = 0
    for i in range(c.HIST_LEN):
        frame = clip[:, :, 3 * i:3 * (i + 1)]
        next_frame = clip[:, :, 3 * (i + 1):3 * (i + 2)]
        # noinspection PyTypeChecker
        diff += np.sum(np.square(next_frame - frame))

    return diff

def get_full_clips(data_dir, num_clips, num_rec_out=1):
    """
    Loads a batch of random clips from the unprocessed train or test data.

    @param data_dir: The directory of the data to read. Should be either c.TRAIN_DIR or c.TEST_DIR.
    @param num_clips: The number of clips to read.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.

    @return: An array of shape
             [num_clips, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + num_rec_out))].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    clips = np.empty([num_clips,
                      c.FULL_HEIGHT,
                      c.FULL_WIDTH,
                      (3 * (c.HIST_LEN + num_rec_out))])

    # get num_clips random episodes
    ep_dirs = np.random.choice(glob(os.path.join(data_dir, '*')), num_clips)

    # get a random clip of length HIST_LEN + num_rec_out from each episode
    for clip_num, ep_dir in enumerate(ep_dirs):
        ep_frame_paths = sorted(glob(os.path.join(ep_dir, '*')))
        start_index = np.random.choice(len(ep_frame_paths) - (c.HIST_LEN + num_rec_out - 1))
        clip_frame_paths = ep_frame_paths[start_index:start_index + (c.HIST_LEN + num_rec_out)]

        # read in frames
        for frame_num, frame_path in enumerate(clip_frame_paths):
            frame = imread(frame_path, mode='RGB')
            norm_frame = normalize_frames(frame)

            clips[clip_num, :, :, frame_num * 3:(frame_num + 1) * 3] = norm_frame

    return clips


def get_full_tracked_frame(data_dir, image=None):
    """
    Loads a batch of random crop from the unprocessed train or test data.

    @param data_dir: The directory of the data to read. Should be either c.TRAIN_DIR or c.TEST_DIR.
       """
    frame_array = np.empty([c.FULL_HEIGHT,
                            c.FULL_WIDTH,
                            3])

    targets = np.empty(
        [2]
    )

    # get num_clips random episodes
    if image is not None:
        ep_dir = os.path.dirname(image)
    else:
        ep_dir = np.random.choice(glob(os.path.join(data_dir, "*")), 1)[0]


    # Load frame target information
    tgt_path = ep_dir + "-tgt.json"
    tgt_path = tgt_path.replace('/Train/', '/TrackTrain/')
    tgt_path = tgt_path.replace('/Test/', '/TrackTest/')

    with open(tgt_path, 'r') as fp:
        frame_targets = json.load(fp)

    if image is not None:
        frame_path = image
        frame_idx = next(i for i, f in enumerate(frame_targets)
                         if os.path.basename(f[0]) == os.path.basename(image))

    else:
        ep_frame_paths = sorted(glob(os.path.join(ep_dir, '*')))
        frame_idx = np.random.choice(len(ep_frame_paths) - 1)
        frame_path = ep_frame_paths[frame_idx]

    assert(
        os.path.basename(frame_path) ==
        os.path.basename(frame_targets[frame_idx][0])
    )

    frame = imread(frame_path, mode='RGB')
    norm_frame = normalize_frames(frame)
    frame_array[:, :, :] = norm_frame
    targets[:] = [frame_targets[frame_idx][1], frame_targets[frame_idx][2]]

    assert(targets[0] < c.FULL_WIDTH)
    assert(targets[1] < c.FULL_HEIGHT)

    return frame_array, targets, frame_path


def process_clip():
    """
    Gets a clip from the train dataset, cropped randomly to c.TRAIN_HEIGHT x c.TRAIN_WIDTH.

    @return: An array of shape [c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
             A frame sequence with values normalized in range [-1, 1].
    """
    clip = get_full_clips(c.TRAIN_DIR, 1)[0]

    # Randomly crop the clip. With 0.05 probability, take the first crop offered, otherwise,
    # repeat until we have a clip with movement in it.
    take_first = np.random.choice(2, p=[0.95, 0.05])
    cropped_clip = np.empty([c.TRAIN_HEIGHT, c.TRAIN_WIDTH, 3 * (c.HIST_LEN + 1)])
    for i in range(100):  # cap at 100 trials in case the clip has no movement anywhere
        crop_x = np.random.choice(c.FULL_WIDTH - c.TRAIN_WIDTH + 1)
        crop_y = np.random.choice(c.FULL_HEIGHT - c.TRAIN_HEIGHT + 1)
        cropped_clip = clip[crop_y:crop_y + c.TRAIN_HEIGHT, crop_x:crop_x + c.TRAIN_WIDTH, :]

        if take_first or clip_l2_diff(cropped_clip) > c.MOVEMENT_THRESHOLD:
            break

    return cropped_clip


def process_tracking_clip():
    """
    Gets a clip from the train dataset, cropped randomly to c.TRAIN_HEIGHT x c.TRAIN_WIDTH.

    @return: An array of shape [c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
             A frame sequence with values normalized in range [-1, 1].
    """
    frame, tgt, path = get_full_tracked_frame(c.TRAIN_DIR)
    #test_tracking_input(frame, tgt.tolist() + [1.0], rel=False)

    # Randomly crop the clip. With 0.30 probability, take the first crop
    # offered, otherwise, repeat until we have a clip with the target in it
    target_present = np.random.choice(2, p=[0.50, .50])
    cropped_target = np.empty([3], dtype=np.float32)

    if target_present:

        x, y = tgt[0], tgt[1]

        dx = x + np.random.uniform(c.TRAIN_WIDTH - 1)
        dy = y + np.random.uniform(c.TRAIN_HEIGHT - 1)

        if dx > c.FULL_WIDTH:
            dx = c.FULL_WIDTH

        if dy > c.FULL_HEIGHT:
            dy = c.FULL_HEIGHT

        crop_x = int(max(0, dx - c.TRAIN_WIDTH))
        crop_y = int(max(0, dy - c.TRAIN_HEIGHT))

        cropped_clip = frame[crop_y:crop_y + c.TRAIN_HEIGHT,
                             crop_x:crop_x + c.TRAIN_WIDTH,
                       :]

        tgt_x_pct = (tgt[0] - crop_x)/c.TRAIN_WIDTH
        tgt_y_pct = (tgt[1] - crop_y)/c.TRAIN_HEIGHT
        tgt_x_cnf = 1. if 0.0 <= tgt_x_pct <= 1.0 else 0.
        tgt_y_cnf = 1. if 0.0 <= tgt_y_pct <= 1.0 else 0.

        cropped_target[:] = [tgt_x_pct, tgt_y_pct, tgt_x_cnf * tgt_y_cnf]

    else:

        crop_x = np.random.choice(c.FULL_WIDTH - c.TRAIN_WIDTH + 1)
        crop_y = np.random.choice(c.FULL_HEIGHT - c.TRAIN_HEIGHT + 1)
        cropped_clip = frame[crop_y:crop_y + c.TRAIN_HEIGHT,
                             crop_x:crop_x + c.TRAIN_WIDTH, :]

        tgt_x_pct = (tgt[0] - crop_x)/c.TRAIN_WIDTH
        tgt_y_pct = (tgt[1] - crop_y)/c.TRAIN_HEIGHT
        tgt_x_cnf = 1. if 0.0 <= tgt_x_pct <= 1.0 else 0.
        tgt_y_cnf = 1. if 0.0 <= tgt_y_pct <= 1.0 else 0.

        cropped_target[:] = [tgt_x_pct, tgt_y_pct, tgt_x_cnf * tgt_y_cnf]

    #test_tracking_input(cropped_clip, cropped_target)
    return cropped_clip, cropped_target


def get_tracking_memorize_batch():
    """
    Loads c.BATCH_SIZE clips from the database of preprocessed training clips.

    @return: An array of shape
            [c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    """
    clips = np.empty([c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, 3],
                     dtype=np.float32)
    tgts = np.empty([c.BATCH_SIZE, 3])
    for i in range(c.BATCH_SIZE):
        c_path = c.TRK_TRAIN_DIR_CLIPS + '1_c.npz'
        t_path = c_path.replace('_c.npz', '_t.npz')

        clips[i] = np.load(c_path)['arr_0']
        tgts[i] = np.load(t_path)['arr_0']

        assert(clips.shape[3] == 3)

    return clips, tgts


def test_tracking_input(frame, tgt, rel=True):
    xsf = c.TRAIN_WIDTH if rel else 1.0
    ysf = c.TRAIN_HEIGHT if rel else 1.0

    plt.imshow(frame, interpolation='nearest')
    plt.scatter(tgt[0]*xsf, tgt[1]*ysf, marker='o',
                color='b', edgecolor='k',
                label="{:.3f}, {:.3f}, {:.3f}".format(*tgt))
    plt.legend()
    plt.show()



def get_tracking_train_batch():
    """
    Loads c.BATCH_SIZE clips from the database of preprocessed training clips.

    @return: An array of shape
            [c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    """
    clips = np.empty([c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, 3],
                     dtype=np.float32)
    tgts = np.empty([c.BATCH_SIZE, 3])
    for i in range(c.BATCH_SIZE):
        c_path = c.TRK_TRAIN_DIR_CLIPS + \
                 str(np.random.choice(c.TRK_NUM_CLIPS)) + '_c.npz'
        t_path = c_path.replace('_c.npz', '_t.npz')

        clips[i] = np.load(c_path)['arr_0']
        tgts[i] = np.load(t_path)['arr_0']

        #test_tracking_input(clips[i], tgts[i])

        assert(clips.shape[3] == 3)

    return clips, tgts


def get_train_batch():
    """
    Loads c.BATCH_SIZE clips from the database of preprocessed training clips.

    @return: An array of shape
            [c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    """
    clips = np.empty([c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))],
                     dtype=np.float32)
    for i in range(c.BATCH_SIZE):
        path = c.TRAIN_DIR_CLIPS + str(np.random.choice(c.NUM_CLIPS)) + '.npz'
        clip = np.load(path)['arr_0']

        clips[i] = clip

    return clips

def get_tracking_test_batch(batch_size, image=None):
    img_crop_info = []
    img_crops = []
    img_crop_tgts = []

    _imgs, _img_tgts, _img_path = get_full_tracked_frame(c.TRAIN_DIR,
                                                         image=image)
    img, img_tgt = _imgs, _img_tgts

    batch_index = 0
    info_batch = np.zeros([c.BATCH_SIZE, 2])
    img_batch = np.zeros([c.BATCH_SIZE, c.TRAIN_HEIGHT,
                          c.TRAIN_WIDTH, 3], dtype=np.float32)
    tgt_batch = np.zeros([c.BATCH_SIZE, 3])

    for crop_x in range(0, c.FULL_WIDTH, c.TRAIN_WIDTH):
        if crop_x > c.FULL_WIDTH - c.TRAIN_WIDTH:
            crop_x = c.FULL_WIDTH - c.TRAIN_WIDTH

        for crop_y in range(0, c.FULL_HEIGHT, c.TRAIN_HEIGHT):
            if crop_y > c.FULL_HEIGHT - c.TRAIN_HEIGHT:
                crop_y = c.FULL_HEIGHT - c.TRAIN_HEIGHT

            crop_info = [crop_x, crop_y]
            cropped_clip = img[crop_y:crop_y + c.TRAIN_HEIGHT,
                               crop_x:crop_x + c.TRAIN_WIDTH, :]

            tgt_x_pct = (img_tgt[0] - crop_x)/c.TRAIN_WIDTH
            tgt_y_pct = (img_tgt[1] - crop_y)/c.TRAIN_HEIGHT
            tgt_x_cnf = 1. if 0.0 <= tgt_x_pct <= 1.0 else 0.
            tgt_y_cnf = 1. if 0.0 <= tgt_y_pct <= 1.0 else 0.

            cropped_target = [tgt_x_pct, tgt_y_pct, tgt_x_cnf * tgt_y_cnf]

            if batch_index < batch_size:
                info_batch[batch_index] = crop_info
                img_batch[batch_index] = cropped_clip
                tgt_batch[batch_index] = cropped_target

            if batch_index >= batch_size:
                img_crop_info.append(info_batch)
                img_crops.append(img_batch)
                img_crop_tgts.append(tgt_batch)

                batch_index = 0
                info_batch = np.zeros([c.BATCH_SIZE, 2])
                img_batch = np.zeros([c.BATCH_SIZE, c.TRAIN_HEIGHT,
                                      c.TRAIN_WIDTH, 3], dtype=np.float32)
                tgt_batch = np.zeros([c.BATCH_SIZE, 3])

                info_batch[batch_index] = crop_info
                img_batch[batch_index] = cropped_clip
                tgt_batch[batch_index] = cropped_target

            batch_index += 1

    # Fill the remaining elements in this batch with random crops!
    for i in range(batch_index, batch_size):
        crop_x = np.random.choice(c.FULL_WIDTH - c.TRAIN_WIDTH + 1)
        crop_y = np.random.choice(c.FULL_HEIGHT - c.TRAIN_HEIGHT + 1)
        crop_info = [crop_x, crop_y]
        cropped_clip = img[crop_y:crop_y + c.TRAIN_HEIGHT,
                           crop_x:crop_x + c.TRAIN_WIDTH, :]

        tgt_x_pct = (img_tgt[0] - crop_x) / c.TRAIN_WIDTH
        tgt_y_pct = (img_tgt[1] - crop_y) / c.TRAIN_HEIGHT
        tgt_x_cnf = 1. if 0.0 <= tgt_x_pct <= 1. else 0.
        tgt_y_cnf = 1. if 0.0 <= tgt_y_pct <= 1. else 0.

        cropped_target = [tgt_x_pct, tgt_y_pct, tgt_x_cnf * tgt_y_cnf]

        info_batch[i] = crop_info
        img_batch[i] = cropped_clip
        tgt_batch[i] = cropped_target

    img_crop_info.append(info_batch)
    img_crops.append(img_batch)
    img_crop_tgts.append(tgt_batch)

    return img_crop_info, img_crops, img_crop_tgts, _img_path, img_tgt


def get_tracking_prod_batch(batch_size, image):
    img_crop_info = []
    img_crops = []

    norm_frame = normalize_frames(image)
    frame_array = norm_frame

    batch_index = 0
    info_batch = np.zeros([c.BATCH_SIZE, 2])
    img_batch = np.zeros([c.BATCH_SIZE, c.TRAIN_HEIGHT,
                          c.TRAIN_WIDTH, 3], dtype=np.float32)

    for crop_x in range(0, c.FULL_WIDTH, c.TRAIN_WIDTH):
        if crop_x > c.FULL_WIDTH - c.TRAIN_WIDTH:
            crop_x = c.FULL_WIDTH - c.TRAIN_WIDTH

        for crop_y in range(0, c.FULL_HEIGHT, c.TRAIN_HEIGHT):
            if crop_y > c.FULL_HEIGHT - c.TRAIN_HEIGHT:
                crop_y = c.FULL_HEIGHT - c.TRAIN_HEIGHT

            crop_info = [crop_x, crop_y]
            cropped_clip = frame_array[crop_y:crop_y + c.TRAIN_HEIGHT,
                                       crop_x:crop_x + c.TRAIN_WIDTH, :]

            if batch_index < batch_size:
                info_batch[batch_index] = crop_info
                img_batch[batch_index] = cropped_clip

            if batch_index >= batch_size:
                img_crop_info.append(info_batch)
                img_crops.append(img_batch)

                batch_index = 0
                info_batch = np.zeros([c.BATCH_SIZE, 2])
                img_batch = np.zeros([c.BATCH_SIZE, c.TRAIN_HEIGHT,
                                      c.TRAIN_WIDTH, 3], dtype=np.float32)

                info_batch[batch_index] = crop_info
                img_batch[batch_index] = cropped_clip

            batch_index += 1

    # Fill the remaining elements in this batch with random crops!
    for i in range(batch_index, batch_size):
        crop_x = np.random.choice(c.FULL_WIDTH - c.TRAIN_WIDTH + 1)
        crop_y = np.random.choice(c.FULL_HEIGHT - c.TRAIN_HEIGHT + 1)
        info_batch[i] = [crop_x, crop_y]
        img_batch[i] = frame_array[crop_y:crop_y + c.TRAIN_HEIGHT,
                                   crop_x:crop_x + c.TRAIN_WIDTH, :]

    img_crop_info.append(info_batch)
    img_crops.append(img_batch)

    return img_crop_info, img_crops


def get_test_batch(test_batch_size, num_rec_out=1):
    """
    Gets a clip from the test dataset.

    @param test_batch_size: The number of clips.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.

    @return: An array of shape:
             [test_batch_size, c.TEST_HEIGHT, c.TEST_WIDTH, (3 * (c.HIST_LEN + num_rec_out))].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    return get_full_clips(c.TEST_DIR, test_batch_size, num_rec_out=num_rec_out)


##
# Error calculation
##

# TODO: Add SSIM error http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf
# TODO: Unit test error functions.

def psnr_error(gen_frames, gt_frames):
    """
    Computes the Peak Signal to Noise Ratio error between the generated images and the ground
    truth images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.

    @return: A scalar tensor. The mean Peak Signal to Noise Ratio error over each frame in the
             batch.
    """
    shape = tf.shape(gen_frames)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])
    square_diff = tf.square(gt_frames - gen_frames)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(square_diff, [1, 2, 3])))
    return tf.reduce_mean(batch_errors)

def sharp_diff_error(gen_frames, gt_frames):
    """
    Computes the Sharpness Difference error between the generated images and the ground truth
    images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.

    @return: A scalar tensor. The Sharpness Difference error over each frame in the batch.
    """
    shape = tf.shape(gen_frames)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])

    # gradient difference
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    # TODO: Could this be simplified with one filter [[-1, 2], [0, -1]]?
    pos = tf.constant(np.identity(3), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding))

    gen_grad_sum = gen_dx + gen_dy
    gt_grad_sum = gt_dx + gt_dy

    grad_diff = tf.abs(gt_grad_sum - gen_grad_sum)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(grad_diff, [1, 2, 3])))
    return tf.reduce_mean(batch_errors)


def create_smart_saver(weights, mute_variables):
    r"""
    By default, TensorFlow's Saver loads all variables in a checkpoint file
    to all variables in the current session. This creates a Saver that will
    only load a weight if it's asked for by the current session, AND not load
    weights that aren't asked for by the current session.
    Parameters
    ----------
    weights : str
        Path to initial weights file.
    mute_variables : list[str]
        List of variables to ignore when loading from the weight file.
        Use when you change the shape of a layer, and don't want to use
        the weights for that layer from a weight file.
    Returns
    -------
    saver : tf.train.Saver
        TensorFlow Saver that won't run into extra/missing variables problems.
    """
    saved_variables = tf.contrib.framework.list_variables(weights)
    desired_variables = tf.global_variables()
    saved_names = set([n for n, _ in saved_variables])
    desired_names = set([_clean_name(v.name) for v in desired_variables])

    true_mute = set()
    for m in mute_variables:
        m = _clean_name(m)
        for d in desired_names:
            v = re.match(m, d)
            if v is not None:
                true_mute.add(v.string)

    clean_mute = [_clean_name(n) for n in true_mute]
    desired_names = desired_names.difference(clean_mute)
    matched_names = saved_names.intersection(desired_names)
    for name in desired_names:
        if name not in matched_names:
            print("Tried to load {variable} but not found in {weights}."
                        " Using default initializer.".format(variable=name,
                                                             weights=weights))
    for name in saved_names:
        if name not in matched_names:
            print("{variable} from weight file ignored.".format(
                variable=name))
    final_variables = [v for v in tf.global_variables()
                       if _clean_name(v.name) in matched_names]

    return tf.train.Saver(var_list=final_variables)


def _clean_name(name):
    r"""
    Gets rid of :0 at the end of TensorFlow variable names.
    Parameters
    ----------
    name : str
        TensorFlow variable name.
    Returns
    -------
    clean_name : str
        TensorFlow variable name without ":0" at the end
    """
    if re.search(':0$', name):
        return name[:-2]
    return name