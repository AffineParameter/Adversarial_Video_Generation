import tensorflow as tf
import numpy as np
from skimage.transform import resize

from detect_scale_model import DetScaleModel
from loss_functions import cat_loss, pos_x_loss, pos_y_loss, trk_loss
from tfutils import w, b
import constants as c
from utils import get_tracking_test_batch
from matplotlib import pylab as plt
import cv2
import os
from glob import glob


# noinspection PyShadowingNames
class DetectionModel:
    def __init__(self, session, summary_writer, height, width, scale_conv_layer_fms,
                 scale_kernel_sizes, scale_fc_layer_sizes):
        """
        Initializes a DetectionModel.

        @param session: The TensorFlow session.
        @param summary_writer: The writer object to record TensorBoard summaries
        @param height: The height of the input images.
        @param width: The width of the input images.
        @param scale_conv_layer_fms: The number of feature maps in each convolutional layer of each
                                     scale network.
        @param scale_kernel_sizes: The size of the kernel for each layer of each scale network.
        @param scale_fc_layer_sizes: The number of nodes in each fully-connected layer of each scale
                                     network.

        @type session: tf.Session
        @type summary_writer: tf.train.SummaryWriter
        @type height: int
        @type width: int
        @type scale_conv_layer_fms: list<list<int>>
        @type scale_kernel_sizes: list<list<int>>
        @type scale_fc_layer_sizes: list<list<int>>
        """
        self.sess = session
        self.summary_writer = summary_writer
        self.height = height
        self.width = width
        self.scale_conv_layer_fms = scale_conv_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.scale_fc_layer_sizes = scale_fc_layer_sizes
        self.num_scale_nets = len(scale_conv_layer_fms)

        self.define_graph()

    # noinspection PyAttributeOutsideInit
    def define_graph(self):
        """
        Sets up the model graph in TensorFlow.
        """
        with tf.name_scope('discriminator'):
            ##
            # Setup scale networks. Each will make the predictions for images at a given scale.
            ##
            self.scale_nets = []
            for scale_num in range(self.num_scale_nets):
                with tf.name_scope('scale_net_' + str(scale_num)):
                    scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                    self.scale_nets.append(DetScaleModel(scale_num,
                                                       int(self.height * scale_factor),
                                                       int(self.width * scale_factor),
                                                       self.scale_conv_layer_fms[scale_num],
                                                       self.scale_kernel_sizes[scale_num],
                                                       self.scale_fc_layer_sizes[scale_num]))

            # A list of the final outputs for the model
            self.scale_last_fc = []
            for scale_num in range(self.num_scale_nets):
                self.scale_last_fc.append(self.scale_nets[scale_num].last_fc)

            # fully-connected
            with tf.name_scope('setup'):
                with tf.name_scope('final-fully-connected'):
                    # Add in a final layer to go from the last scale fcs to a
                    # single output
                    ffc_ws = []
                    ffc_bs = []
                    for i, scale_fc in enumerate(self.scale_last_fc):
                        ffc_ws.append(w((self.scale_fc_layer_sizes[i][-1],
                                         c.FINAL_FC_LAYER_SIZES_DET[0])))
                        ffc_bs.append(b((c.FINAL_FC_LAYER_SIZES_DET[0],)))

                    ffc_ws.append(w((c.FINAL_FC_LAYER_SIZES_DET[0],
                                     c.FINAL_FC_LAYER_SIZES_DET[1])))
                    ffc_bs.append(b((c.FINAL_FC_LAYER_SIZES_DET[1],)))

            # fully-connected layers
            with tf.name_scope('calculate'):
                with tf.name_scope('final-fully-connected'):
                    nl_fc = tf.zeros([self.scale_nets[0].batch_size,
                                      c.FINAL_FC_LAYER_SIZES_DET[0]])

                    for i, scale_fc in enumerate(self.scale_last_fc):
                        nl_fc = tf.add(nl_fc,
                                       tf.matmul(scale_fc, ffc_ws[i]) +
                                       ffc_bs[i])

                    # Next To last Layer
                    nl_fca = tf.nn.relu(nl_fc)

                    # Last Layer
                    ll_fc = tf.matmul(nl_fca, ffc_ws[-1]) + ffc_bs[-1]
                    self.preds = tf.nn.sigmoid(ll_fc)

            ##
            # Data
            ##
            self.labels = tf.placeholder(tf.float32, shape=[None, 3],
                                         name='labels')

            ##
            # Training
            ##
            with tf.name_scope('training'):
                # global loss is the combined loss from every scale network
                self.global_loss = trk_loss(self.preds, self.labels,
                                            wgt_cat=c.DET_WGT_CAT,
                                            wgt_pos=c.DET_WGT_POS)
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                self.optimizer = tf.train.GradientDescentOptimizer(
                    c.LRATE_DET, name='optimizer')
                self.train_op = self.optimizer.minimize(self.global_loss,
                                                        global_step=self.global_step,
                                                        name='train_op')

                with tf.name_scope('loss'):
                    # error computation
                    # get error at largest scale
                    self.cat_loss = cat_loss(self.preds, self.labels)
                    self.pos_x_loss = pos_x_loss(self.preds, self.labels)
                    self.pos_y_loss = pos_y_loss(self.preds, self.labels)
                    self.global_loss = trk_loss(self.preds, self.labels,
                                                wgt_cat=c.DET_WGT_CAT,
                                                wgt_pos=c.DET_WGT_POS)
                    self.summaries = [
                        tf.summary.scalar('cat_loss', self.cat_loss),
                        tf.summary.scalar('pos_x_loss', self.pos_x_loss),
                        tf.summary.scalar('pos_y_loss', self.pos_y_loss),
                        tf.summary.scalar('global_loss', self.global_loss)
                    ]
            # add summaries to visualize in TensorBoard
            self.summaries = tf.summary.merge(self.summaries)

            ##
            # Training
            ##
            with tf.name_scope('test'):
                with tf.name_scope('loss'):
                    # error computation
                    # get error at largest scale
                    self.test_cat_loss = cat_loss(self.preds, self.labels)
                    self.test_pos_x_loss = pos_x_loss(self.preds, self.labels)
                    self.test_pos_y_loss = pos_y_loss(self.preds, self.labels)
                    self.test_global_loss = trk_loss(self.preds, self.labels,
                                                     wgt_cat=c.DET_WGT_CAT,
                                                     wgt_pos=c.DET_WGT_POS)
                    self.test_summaries = [
                        tf.summary.scalar('cat_loss', self.test_cat_loss),
                        tf.summary.scalar('pos_x_loss', self.test_pos_x_loss),
                        tf.summary.scalar('pos_y_loss', self.test_pos_y_loss),
                        tf.summary.scalar('global_loss', self.test_global_loss)
                    ]

            # add summaries to visualize in TensorBoard
            self.test_summaries = tf.summary.merge(self.test_summaries)


    def build_feed_dict(self, input_frames, targets):
        """
        Builds a feed_dict with resized inputs and outputs for each scale network.

        @param input_frames: An array of shape
                             [batch_size x self.height x self.width x (3 * HIST_LEN)], 
                             The frames to use for detection.
        @param targets: An array of shape [batch_size x 3], 
                            The ground truth outputs for each sequence in 
                            input_frames. (x, y, confidence)

        @return: The feed_dict needed to run this network, all scale_nets, and the generator
                 predictions.
        """
        feed_dict = {}
        batch_size = np.shape(input_frames)[0]

        ##
        # Create detector feed dict
        ##
        for scale_num in range(self.num_scale_nets):
            scale_net = self.scale_nets[scale_num]

            # resize input_frames
            scaled_input_frames = np.empty([batch_size, scale_net.height,
                                            scale_net.width, 3])
            for i, clip in enumerate(input_frames):
                # for skimage.transform.resize, images need to be in range [0, 1], so normalize to
                # [0, 1] before resize and back to [-1, 1] after
                sknorm_img = (clip / 2) + 0.5
                resized_frame = resize(sknorm_img, [scale_net.height,
                                                    scale_net.width,
                                                    3])
                scaled_input_frames[i] = (resized_frame - 0.5) * 2

            # convert to np array and add to feed_dict
            feed_dict[scale_net.input_frames] = scaled_input_frames

        # add labels for each image to feed_dict
        batch_size = np.shape(input_frames)[0]
        feed_dict[self.labels] = targets

        return feed_dict


    def train_step(self, batch_c, batch_t):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch_c: An array of shape
                      [BATCH_SIZE x self.height x self.width x (3 * HIST_LEN)]. 
                      The input frames, concatenated along the channel axis (
                      index 3).
        @param batch_t: The targets.

        @return: The global step.
        """
        ##
        # Train
        ##
        feed_dict = self.build_feed_dict(batch_c, batch_t)

        _, global_loss, cat_loss, pos_x_loss, \
        pos_y_loss, global_step, summaries = \
            self.sess.run(
            [self.train_op,
             self.global_loss,
             self.cat_loss,
             self.pos_x_loss,
             self.pos_y_loss,
             self.global_step,
             self.summaries],
            feed_dict=feed_dict)

        ##
        # User output
        ##

        if global_step % c.STATS_FREQ == 0:
            min_dr_n = self.get_pix_accuracy(c.TRAIN_DIR, top_n=3)
            print('DetectionModel: step %d | global loss: %f' % (global_step, global_loss))
            print('  cat loss     : %f' % (cat_loss,))
            print('  pos x loss   : %f' % (pos_x_loss,))
            print('  pos y loss   : %f' % (pos_y_loss,))
            print('  pix acc_top1 : %f' % (min_dr_n[0],))
            print('  pix acc_top2 : %f' % (min_dr_n[1],))
            print('  pix acc_top3 : %f' % (min_dr_n[2],))
        if global_step % c.SUMMARY_FREQ == 0:
            print('DetectionModel: saved summaries')
            self.summary_writer.add_summary(summaries, global_step)
        if global_step % c.IMG_SAVE_FREQ == 0:

            ep_dir = np.random.choice(glob(os.path.join(c.TEST_DIR, "*")), 1)[0]
            save_dir = c.get_dir(os.path.join(c.IMG_SAVE_DIR,
                                              'Step_' + str(global_step),
                                              "imgs")
                                 )
            fig = plt.figure()
            ax = plt.gca()
            files = sorted(glob(os.path.join(ep_dir, '*')))
            r = np.random.choice(len(files) - c.NUM_FRAMES_PER_CLIP)
            for f in files[r:r+c.NUM_FRAMES_PER_CLIP]:
                self.gen_image(ax, f, save_dir)
            plt.close(fig)
            print("Images Saved!...")

        return global_step


    def test_step(self, global_step):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch_c: An array of shape
                      [BATCH_SIZE x self.height x self.width x (3 * HIST_LEN)]. 
                      The input frames, concatenated along the channel axis (
                      index 3).
        @param batch_t: The targets.

        @return: The global step.
        """
        ##
        # Train
        ##
        img_crop_info, img_crops, img_crop_tgts, _img_path, _img_tgt = \
            get_tracking_test_batch(c.BATCH_SIZE)

        feed_dict = self.build_feed_dict(img_crops[0], img_crop_tgts[0])

        summaries, global_loss, cat_loss, \
        pos_x_loss, pos_y_loss = self.sess.run(
            [self.test_summaries,
             self.test_global_loss,
             self.test_cat_loss,
             self.test_pos_x_loss,
             self.test_pos_y_loss],
            feed_dict=feed_dict)

        ##
        # User output
        ##
        min_dr_n = self.get_pix_accuracy(c.TRAIN_DIR, top_n=3)
        print('Test DetectionModel: step %d | global loss: %f' % (
               global_step,  global_loss))
        print('  cat loss     : %f' % (cat_loss,))
        print('  pos x loss   : %f' % (pos_x_loss,))
        print('  pos y loss   : %f' % (pos_y_loss,))
        print('  pix acc_top1 : %f' % (min_dr_n[0],))
        print('  pix acc_top2 : %f' % (min_dr_n[1],))
        print('  pix acc_top3 : %f' % (min_dr_n[2],))
        self.summary_writer.add_summary(summaries, global_step)

    def test_image(self, img, top_n=1):

        img_crop_info, img_crops, img_crop_tgts, _img_path, _img_tgt = \
            get_tracking_test_batch(c.BATCH_SIZE, image=img)

        targets = []
        for i in range(len(img_crop_tgts)):
            feed_dict = self.build_feed_dict(img_crops[i], img_crop_tgts[i])
            preds = self.sess.run([self.preds], feed_dict=feed_dict)[0]
            for j, p in enumerate(preds):
                targets.append(
                    (
                        p[0] * c.TRAIN_WIDTH + img_crop_info[i][j][0],
                        p[1] * c.TRAIN_HEIGHT + img_crop_info[i][j][1],
                        p[2],  # Confidence
                    )
                )

        return sorted(targets, key=lambda x: x[2])[:top_n], _img_tgt, img_crop_info

    def get_pix_accuracy(self, dir, top_n=1):
        ep_dir = np.random.choice(glob(os.path.join(dir, "*")), 1)[0]
        files = sorted(glob(os.path.join(ep_dir, '*')))
        r = np.random.choice(len(files) - c.NUM_TEST_FRAMES)
        results = []
        for f in files[r:r + c.NUM_TEST_FRAMES]:
            results.append(self.test_image(f, top_n=top_n)[0:2])

        mean_distances = [0] * top_n
        for i in range(top_n):
            for pred, tgt in results:
                _pred = np.array(pred)
                dx2 = np.min((_pred[0:i+1, 0] - tgt[0])**2)
                dy2 = np.min((_pred[0:i+1, 1] - tgt[1])**2)
                mean_distances[i] += np.sqrt(dx2 + dy2)

            mean_distances[i] /= c.NUM_TEST_FRAMES

        return mean_distances

    def gen_image(self, ax, img, dir):

        targets, _img_tgt, img_crops = self.test_image(img, top_n=3)
        img_bgr = cv2.imread(img, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.clear()
        ax.imshow(img_rgb, interpolation='nearest')
        # ax.scatter([i[1] for b in img_crops for i in b],
        #            [i[0] for b in img_crops for i in b],
        #            marker='+', color='yellow')
        ax.scatter(_img_tgt[0], _img_tgt[1],
                   marker='o', color='blue', edgecolor='black',
                   label="GT")
        ax.scatter(targets[0][0], targets[0][1],
                   marker='x', color='green',
                   label="C:{:.2f}".format(targets[0][2]))
        ax.scatter(targets[1][0], targets[1][1],
                   marker='x', color='orange',
                   label="C:{:.2f}".format(targets[1][2]))
        ax.scatter(targets[2][0], targets[2][1],
                   marker='x', color='red',
                   label="C:{:.2f}".format(targets[2][2]))
        ax.legend()

        basename, ext = os.path.splitext(img)
        name = os.path.basename(basename)
        out_path = os.path.join(dir, "trk_" + name + ".png")
        print("Saving Figure: {}".format(out_path))
        plt.gcf().savefig(out_path)

        # get num_clips random episodes
