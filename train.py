from pipeline import *
import argparse
from data_loarder import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Glow: Generative Flow with Invertible 1x1 Convolutions",
                                     description="My implementation of GLOW from the paper https://arxiv.org/pdf/1807.03039 in Tensorflow 2")
    parser.add_argument('--k_glow', type=int, nargs='?', default=K_GLOW,
                        help='The amount of blocks per layer')
    parser.add_argument('--l_glow', type=int, nargs='?', default=L_GLOW,
                        help='The amount of layers')
    parser.add_argument('--window_size', type=int, nargs='?', default=WINDOW_SIZE,
                        help='The window length in seconds of the input sound data (this is dataset dependent, should short than the length of input data)')
    parser.add_argument('--sampling_rate', type=int, nargs='?', default=SAMPLING_RATE,
                        help='The sampling rate of the input sound data (this is dataset dependent)')
    parser.add_argument('--channel_size', type=int, nargs='?', default=CHANNEL_SIZE,
                        help='The channel size of the input sound data, default value is 1 (this is dataset dependent)')

    args = parser.parse_args()
    K_GLOW = args.k_glow
    L_GLOW = args.l_glow
    WINDOW_SIZE = args.window_size
    SAMPLING_RATE = args.sampling_rate
    CHANNEL_SIZE = args.channel_size

    WINDOW_LENGTH = int(SAMPLING_RATE * WINDOW_SIZE)

    parser.print_help()  # print the help of the parser

    # Step 1. the data, split between train and test sets
    data_loader = SongDataLoader('real.tfrecords'
                                 , tfrecord_dir=r'D:\PlayGround\research\SinGlow\runs')
    data_loader.make(r'D:\PlayGround\research\SongDatabase\RealSinger\vocal collection\wav files')
    train_dataset = data_loader.load(sampling_num=200)

    # Step 2. the brain
    brain = Brain(SQUEEZE_FACTOR, K_GLOW, L_GLOW, WINDOW_LENGTH, CHANNEL_SIZE, LEARNING_RATE)

    # Step 3. training iteration

    # define metrics variables
    nll = tf.keras.metrics.Mean("nll")
    mean_z_squared = tf.keras.metrics.Mean("mean_z_squared")
    var_z = tf.keras.metrics.Mean("var_z")

    # TENSORBOARD
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = TENSORBOARD_LOGDIR + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    _toggle_training = False
    for ep in range(EPOCHS):
        print(f"epoch {ep + 1}/{EPOCHS}")
        nll.reset_states()
        mean_z_squared.reset_states()
        var_z.reset_states()

        # iteration per epoch
        with tqdm(enumerate(train_dataset), total=data_loader.batch_number) as t:
            for i, x_t in t:
                if _toggle_training:
                    z, _nll_x = brain.train_step(x_t)  # run the train step and store the nll in the variable
                    mean_z_squared(tf.reduce_mean(z, axis=-1) ** 2)
                    var_z(tf.math.reduce_variance(z, axis=-1))
                    nll(_nll_x)
                    t.set_postfix(nll=nll.result().numpy(), mean_sq=mean_z_squared.result().numpy(),
                                  var=var_z.result().numpy())
                else:  # to initiate some variables necessary
                    brain.model(x_t, training=True)
                    if LOAD_WEIGHT: print(brain.load_weights(CHECKPOINT_PATH))
                    _toggle_training = True
                if i + 1 % 200 == 0:
                    # save weight every 2000 batchs
                    brain.save_weights(CHECKPOINT_PATH)

        # save weight every epoch
        brain.save_weights(CHECKPOINT_PATH)

        # TENSORBOARD Save
        with train_summary_writer.as_default():
            tf.summary.scalar('nll', nll.result(), step=ep)
            tf.summary.scalar('mean_sq', mean_z_squared.result(), step=ep)
            tf.summary.scalar('var', var_z.result(), step=ep)
