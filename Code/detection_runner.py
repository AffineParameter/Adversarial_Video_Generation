import tensorflow as tf
import getopt
import sys
import os

from utils import get_tracking_train_batch, get_tracking_test_batch, \
    create_smart_saver, get_tracking_memorize_batch
import constants as c
from detect_model import DetectionModel

class DetectionRunner:
    def __init__(self, num_steps, model_load_path):
        """
        Initializes the Adversarial Video Generation Runner.

        @param num_steps: The number of training steps to run.
        @param model_load_path: The path from which to load a previously-saved model.
                                Default = None.
        """

        self.global_step = 0
        self.num_steps = num_steps

        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter(c.SUMMARY_SAVE_DIR,
                                                    graph=self.sess.graph)

        print('Init detector...')
        self.detect_model = DetectionModel(self.sess,
                                           self.summary_writer,
                                           c.TRAIN_HEIGHT,
                                           c.TRAIN_WIDTH,
                                           c.SCALE_CONV_FMS_DET,
                                           c.SCALE_KERNEL_SIZES_DET,
                                           c.SCALE_FC_LAYER_SIZES_DET)

        print('Init variables...')
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.sess.run(tf.global_variables_initializer())

        # if load path specified, load a saved model
        if model_load_path is not None:
            smart_saver = create_smart_saver(model_load_path,
                                             mute_variables=[])
            smart_saver.restore(self.sess, model_load_path)
            print('Model restored from ' + model_load_path)

    def train(self):
        """
        Runs a training loop on the model networks.
        """
        for i in range(self.num_steps):
            # update discriminator
            batch_c, batch_t = get_tracking_train_batch()
            self.global_step = self.detect_model.train_step(batch_c, batch_t)

            # save the models
            if self.global_step % c.MODEL_SAVE_FREQ == 0:
                print('-' * 30)
                print('Saving {}-iteration models...'.format(self.global_step))
                self.saver.save(self.sess,
                                c.MODEL_SAVE_DIR + 'detect_model.ckpt',
                                global_step=self.global_step)
                print('Saved models!')
                print('-' * 30)

            # test generator model
            if self.global_step % c.TEST_FREQ == 0:
                self.test()

    def test(self):
        """
        Runs one test step on the generator network.
        """
        self.detect_model.test_step(self.global_step)

    def memorize(self):
        """
        Runs a training loop on the model networks.
        """
        for i in range(self.num_steps):
            # update discriminator
            batch_c, batch_t = get_tracking_memorize_batch()
            self.global_step = self.detect_model.train_step(batch_c, batch_t)

            # save the models
            if self.global_step % c.MODEL_SAVE_FREQ == 0:
                print('-' * 30)
                print('Saving {}-iteration models...'.format(self.global_step))
                self.saver.save(self.sess,
                                c.MODEL_SAVE_DIR + 'detect_model.ckpt',
                                global_step=self.global_step)
                print('Saved models!')
                print('-' * 30)

            # test generator model
            if self.global_step % c.TEST_FREQ == 0:
                self.test()

    def track(self, in_dir, out_dir):

        self.detect_model.render_production_images(in_dir, out_dir, lim=120)



def usage():
    print('Options:')
    print('-l/--load_path=    <Relative/path/to/saved/model>')
    print('-t/--test_dir=     <Directory of test images>')
    print('-r/--recursions=   <# recursive predictions to make on test>')
    print('-a/--adversarial=  <{t/f}> (Whether to use adversarial training. Default=True)')
    print('-n/--name=         <Subdirectory of ../Data/Save/*/ in which to save output of this run>')
    print('-s/--steps=        <Number of training steps to run> (Default=1000001)')
    print('-O/--overwrite     (Overwrites all previous data for the model with this save name)')
    print('-T/--test_only     (Only runs a test step -- no training)')
    print('-H/--help          (Prints usage)')
    print('--stats_freq=      <How often to print loss/train error stats, in # steps>')
    print('--summary_freq=    <How often to save loss/error summaries, in # steps>')
    print('--img_save_freq=   <How often to save generated images, in # steps>')
    print('--test_freq=       <How often to test the model on test data, in # steps>')
    print('--model_save_freq= <How often to save the model, in # steps>')


def main():
    ##
    # Handle command line input.
    ##

    load_path = None
    test_only = False
    memorize = False
    num_steps = 1000001
    render_in = render_out = None
    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'l:t:r:a:n:s:OTH',
                                ['load_path=', 'test_dir=', 'recursions=', 'adversarial=', 'name=',
                                 'steps=', 'overwrite', 'test_only', 'help', 'stats_freq=',
                                 'summary_freq=', 'img_save_freq=', 'test_freq=',
                                 'model_save_freq=', 'memorize', 'render_in=',
                                 'render_out='])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-l', '--load_path'):
            load_path = arg
        if opt in ('-t', '--test_dir'):
            c.set_test_dir(arg)
        if opt in ('-f', '--frames'):
            c.NUM_FRAMES_PER_CLIP = int(arg)
        if opt in ('-a', '--adversarial'):
            c.ADVERSARIAL = (arg.lower() == 'true' or arg.lower() == 't')
        if opt in ('-n', '--name'):
            c.set_save_name(arg)
        if opt in ('-s', '--steps'):
            num_steps = int(arg)
        if opt in ('-O', '--overwrite'):
            c.clear_save_name()
        if opt in ('-H', '--help'):
            usage()
            sys.exit(2)
        if opt in ('-T', '--test_only'):
            test_only = True
        if opt == '--stats_freq':
            c.STATS_FREQ = int(arg)
        if opt == '--summary_freq':
            c.SUMMARY_FREQ = int(arg)
        if opt == '--img_save_freq':
            c.IMG_SAVE_FREQ = int(arg)
        if opt == '--test_freq':
            c.TEST_FREQ = int(arg)
        if opt == '--model_save_freq':
            c.MODEL_SAVE_FREQ = int(arg)
        if opt == '--memorize':
            memorize = True
        if opt == '--render_in':
            render_in = arg
        if opt == '--render_out':
            render_out = arg




    ##
    # Init and run the predictor
    ##
    runner = DetectionRunner(num_steps, load_path)

    if memorize:
        # set test frame dimensions
        assert os.path.exists(c.TEST_DIR)
        c.FULL_HEIGHT, c.FULL_WIDTH = c.get_test_frame_dims()
        runner.memorize()

    elif test_only:
        # set test frame dimensions
        assert os.path.exists(c.TEST_DIR)
        c.FULL_HEIGHT, c.FULL_WIDTH = c.get_test_frame_dims()
        runner.test()

    elif render_in and render_out:
        runner.track(render_in, render_out)

    else:
        # set test frame dimensions
        assert os.path.exists(c.TEST_DIR)
        c.FULL_HEIGHT, c.FULL_WIDTH = c.get_test_frame_dims()
        runner.train()


if __name__ == '__main__':
    main()
