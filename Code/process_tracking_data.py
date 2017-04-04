import numpy as np
import getopt
import sys
from glob import glob
import os
import cv2
import json

import constants as c
from utils import process_tracking_clip


def process_image(path, bgr_cuts):
    img_bgr = cv2.imread(path, 1)
    mask = img_bgr[:, :, 0] > -1

    mask = np.logical_and(mask, img_bgr[:, :, 0] >= bgr_cuts[0][0])
    mask = np.logical_and(mask, img_bgr[:, :, 0] <= bgr_cuts[0][1])

    mask = np.logical_and(mask, img_bgr[:, :, 1] >= bgr_cuts[1][0])
    mask = np.logical_and(mask, img_bgr[:, :, 1] <= bgr_cuts[1][1])

    mask = np.logical_and(mask, img_bgr[:, :, 2] >= bgr_cuts[2][0])
    mask = np.logical_and(mask, img_bgr[:, :, 2] <= bgr_cuts[2][1])

    net_p = np.sum(mask[:, :] == True)
    if net_p == 0:
        return [-2, -2]

    y, x = np.mean(np.argwhere(mask[:, :] == True), axis=0).tolist()
    std_y, std_x = np.std(np.argwhere(mask[:, :] == True), axis=0).tolist()

    if std_y < 5.:
        if x < mask.shape[1] / 2:
            mask[:, -50:] = False
        else:
            mask[:, :50] = False

        y, x = np.mean(np.argwhere(mask[:, :] == True), axis=0).tolist()
        std_y, std_x = np.std(np.argwhere(mask[:, :] == True), axis=0).tolist()

    if std_y**2 + std_x**2 > 25.:
        return [-1, -1]

    return [x, y]


def process_images(dataset, outdir):
    for root, clips, frames in os.walk(dataset):
        if len(frames) and frames[0].endswith('.png'):
            # print("Processing Frames From Clip: " + os.path.basename(root))

            frame_target = []
            for frame in frames:
                frame_path = os.path.join(root, frame)

                x, y = process_image(frame_path,
                                     [[70, 150], [163, 165],  [206, 214]])

                frame_target.append(
                    (frame, [x, y])
                )

            frame_target = sorted(frame_target, key=lambda x: x[0])
            f_name = os.path.basename(root + '-tgt.json')
            outpath = os.path.join(outdir, f_name)

            with open(outpath, 'w') as fp:
                json.dump(frame_target, fp)
            print("Targets here: {}".format(outpath))


def get_tracking_targets():
    process_images(c.TEST_DIR, c.TRK_TEST_DIR)
    process_images(c.TRAIN_DIR, c.TRK_TRAIN_DIR)


def process_training_data(num_clips):
    """
    Processes random training clips from the full training data. Saves to TRAIN_DIR_CLIPS by
    default.

    @param num_clips: The number of clips to process. Default = 5000000 (set in __main__).

    @warning: This can take a couple of hours to complete with large numbers of clips.
    """
    num_prev_clips = len(glob(c.TRK_TRAIN_DIR_CLIPS + '*'))

    for frame_num in range(num_prev_clips, num_clips + num_prev_clips):
        frame, tgt = process_tracking_clip()

        np.savez_compressed(c.TRK_TRAIN_DIR_CLIPS + str(frame_num) + '_c', frame)
        np.savez_compressed(c.TRK_TRAIN_DIR_CLIPS + str(frame_num) + '_t', tgt)

        if (frame_num + 1) % 100 == 0:
            print('Processed %d clips' % (frame_num + 1))


def usage():
    print('Options:')
    print('-n/--num_clips= <# clips to process for training> (Default = 5000000)')
    print('-t/--train_dir= <Directory of full training frames>')
    print('-c/--clips_dir= <Save directory for processed clips>')
    print("                (I suggest making this a hidden dir so the filesystem doesn't freeze")
    print("                 with so many files. DON'T `ls` THIS DIRECTORY!)")
    print('-o/--overwrite  (Overwrites the previous data in clips_dir)')
    print('-H/--help       (Prints usage)')


def main():
    ##
    # Handle command line input
    ##

    num_clips = 5000000

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'n:t:c:oH',
                                ['num_clips=', 'train_dir=', 'clips_dir=',
                                 'overwrite', 'track', 'help'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-n', '--num_clips'):
            num_clips = int(arg)
        if opt in ('-t', '--train_dir'):
            c.TRAIN_DIR = c.get_dir(arg)
        if opt in ('-c', '--clips_dir'):
            c.TRK_TRAIN_DIR_CLIPS = c.get_dir(arg)
        if opt in ('-o', '--overwrite'):
            c.clear_dir(c.TRK_TRAIN_DIR_CLIPS)
        if opt in ('-H', '--track'):
            get_tracking_targets()
        if opt in ('-H', '--help'):
            usage()
            sys.exit(2)

    # set train frame dimensions
    assert os.path.exists(c.TRAIN_DIR)
    c.FULL_HEIGHT, c.FULL_WIDTH = c.get_train_frame_dims()

    ##
    # Process data for training
    ##
    process_training_data(num_clips)


if __name__ == '__main__':
    main()
