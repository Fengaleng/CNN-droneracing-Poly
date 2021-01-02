import os
import tensorflow as tf
import pandas as pd
import random
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import add
from tensorflow.keras import regularizers
import time
from training import freeze_network_wts_p1, train_step_1, test_step, unfreeze_network_wts, freeze_network_wts_p2, \
    train_step_2, freeze_network_wts, test_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_model(path_to_h5_file, file_name):
    model_path = os.path.join(path_to_h5_file, file_name)
    model = tf.keras.models.load_model(model_path)

    print('Model loaded successfully!')
    return model


def test_model_speed(model, test_gen):
    # Check speed of algorithm (fps or number of images processed per second)
    start = time.time()
    for i, elem in enumerate(test_gen):

        if i == len(test_gen):
            break

        _, _ = model(elem, training=False)

    duration = time.time() - start
    print(i)
    print(duration)
    fps = (i + 1) / duration
    print("Model speed is approx. {0:.2f} Frames per Second".format(fps))
    return fps


def resnet11(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture of a resnet 11.

    # Arguments
       img_width: Target image width.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2, 2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2, 2])(x1)

    # First residual block
    x2 = BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2, 2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2, 2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x6 = BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2, 2], padding='same')(x5)
    x7 = add([x5, x6])

    # Fourth special residual block
    x8 = BatchNormalization()(x7)
    x8 = Activation('relu')(x8)
    x8 = Conv2D(256, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x8)

    x8 = BatchNormalization()(x8)
    x8 = Activation('relu')(x8)
    x8 = Conv2D(256, (3, 3), strides=[2, 2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x8)

    x8 = BatchNormalization()(x8)
    x8 = Activation('relu')(x8)
    x8 = Conv2D(256, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x8)

    x7 = Conv2D(256, (1, 1), strides=[2, 2], padding='same')(x7)
    x9 = add([x7, x8])

    x = Flatten()(x9)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Position channel
    mean = Dense(256, name='hid_lay_mean_1',
                 kernel_initializer="he_normal",
                 kernel_regularizer=regularizers.l2(1e-4))(x)
    mean_hid = Dense(128, name='hid_lay_mean_2',
                     kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(1e-4))(mean)
    mean_mlp = Dense(output_dim, name='mean')(mean_hid)

    # Covariance channel
    cov = Dense(256, name='hid_lay_cov_1',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x)
    cov_mlp = Dense(output_dim, name='cov',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(cov)
    cov_mlp_pos = Activation(tf.keras.activations.exponential, name='cov_act')(cov_mlp)

    # Define steering-collision model
    model = tf.keras.Model(inputs=[img_input], outputs=[mean_mlp, cov_mlp_pos])
    print(model.summary())

    return model


def training(root_dir, hyperparameters, model, train_metric, val_metric, loss_function, optimizer):
    """
    Performs all the necessary operations for the training of the model
    :param hyperparameters: Dictionary with learning rate, beta 1 and 2, epsilon, batch size
    :param model: Customized ResNet11 model
    :param train_metric: Metric used for the training (MAE by default)
    :param val_metric: Metric used for the validation (MAE by default)
    :param loss_function: Loss function used for the training
    :return:
    """
    # Adjust model weights for first part of training
    model = freeze_network_wts_p1(model)

    training_errors_1 = []
    training_loss_1 = []
    validation_errors_1 = []
    train_start_time_1 = time.time()

    # Prepare the loop brakers in case of overfitting
    overfit_warn = 0
    overfit_break = False
    train_stagn = 0
    train_stagn_break = False

    # ImageDataGenerator creation
    seed_list = []
    image_gen = ImageDataGenerator(rescale=1. / 255)
    new_train_p1_df = pd.read_csv(os.path.join(root_dir, 'new_train_dataset_v2.csv'))

    for epoch in range(hyperparameters["Epoch 1"]):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # K-cross validation manual implementation
        if epoch % 5 == 0 or epoch == 0:
            new_rand_seed = random.randint(0, 100)
            new_train_gen_p1 = image_gen.flow_from_dataframe(dataframe=new_train_p1_df,
                                                             directory=None,
                                                             x_col='Full_path',
                                                             y_col=['distance_rho', 'distance_theta', 'distance_phi',
                                                                    'yaw_diff'],
                                                             target_size=(200, 300),
                                                             color_mode='rgb',
                                                             class_mode='raw',
                                                             batch_size=hyperparameters["Batch Size"],
                                                             shuffle=True,
                                                             seed=new_rand_seed)

            print('New ImageDataGenerators created with seed = {}'.format(new_rand_seed))
            seed_list.append(new_rand_seed)

        # Iterate through training datasets batches
        for i, dataset in enumerate(new_train_gen_p1):
            if i == int(0.8 * len(new_train_gen_p1)):
                break

            # The batches are now randomized; Images in each batch are not
            # print(train_gen_p1[randomlist[i]])
            # print(labels_train_p1[randomlist[i] * batch_size:(randomlist[i] + 1) * batch_size])
            loss_value = train_step_1(dataset, model, loss_function, optimizer, train_metric)

            # Log every 50 batches.
            if i % 50 == 0:
                # loss_value = tf.make_tensor_proto(loss_value)
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (i, float(loss_value))  # tf.make_ndarray(loss_value)
                )
                training_loss_1.append(float(loss_value))  # tf.make_ndarray(loss_value)
                print("Seen so far: %d samples" % ((i + 1) * hyperparameters["Batch Size"]))

        # Display metrics at the end of each epoch and save to list for a later graph
        train_diff = train_metric.result()
        print("Training mean difference over epoch: %.4f" % (float(train_diff)))
        training_errors_1.append(float(train_diff))

        # Reset training metrics at the end of each epoch
        train_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        # for i, dataset in enumerate(new_train_gen_p1):
        for i in range(int(0.2 * len(new_train_gen_p1))):
            if i == int(0.2 * len(new_train_gen_p1) + 1):
                break

            test_step(new_train_gen_p1[i + int(0.8 * len(new_train_gen_p1))], model, val_metric)
            print('Validation {}/{}'.format(i + 1, int(0.2 * len(new_train_gen_p1) + 1)))

        val_acc = val_metric.result()
        val_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc)))
        validation_errors_1.append(float(val_acc))
        print("Time taken: %.2fs" % (time.time() - start_time))

        # Training stagnation warnings (when the model is not improving anymore for 3 times in a row)
        if validation_errors_1[epoch] <= 0.1:  # Usually before that, stagnation are accidental
            if training_errors_1[epoch] - training_errors_1[epoch - 1] > 0.1:
                print("Training stagnation!")

                train_stagn += 1

                print("Warning: Validation stagnation streak, case #{}".format(train_stagn))
                train_stagn_streak = epoch

                if train_stagn == 3:
                    train_stagn_break = True

        if train_stagn_break:
            print("Stopped training because model not improving for 3 cases encountered!")
            print("Last epoch: {}".format(epoch))
            break

        # Overfit warnings and breaking the loop if necessary
        if (training_errors_1[epoch] - validation_errors_1[epoch]) <= -0.01:
            overfit_warn += 1
            print("Warning: Overfit warnings #{}".format(overfit_warn))
            if overfit_warn == 3:
                overfit_break = True

        if overfit_break:
            print("Broke the loop because 3 overfitting cases encountered!")
            print("Last epoch: {}".format(epoch))
            break

    train_time_1 = time.time() - train_start_time_1

    # PART 2 of the training #
    # Prepare the model
    model = freeze_network_wts_p2(unfreeze_network_wts(model))

    training_loss_2 = []
    train_start_time_2 = time.time()

    # Prepare the dataset
    train_p2_df = pd.read_csv(os.path.join(root_dir, 'full_dataset_with_no_gate.csv'))
    new_rand_seed = random.randint(0, 100)
    train_gen_p2 = image_gen.flow_from_dataframe(dataframe=train_p2_df,
                                                 directory=None,
                                                 x_col='Full_path',
                                                 y_col=['distance_rho', 'distance_theta', 'distance_phi', 'yaw_diff'],
                                                 target_size=(200, 300),
                                                 color_mode='rgb',
                                                 class_mode='raw',
                                                 batch_size=hyperparameters["Batch Size"],
                                                 shuffle=True,
                                                 seed=new_rand_seed)
    seed_list.append(new_rand_seed)

    for epoch in range(hyperparameters["Epoch 2"]):
        print("\nStart of 2nd part of training, epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate through training datasets batches
        for i, dataset in enumerate(train_gen_p2):
            # if i == len(train_gen_p2)-1:
            #   loss_value = train_step_2(dataset, labels_train_p2[i * batch_size:])

            if i == len(train_gen_p2):
                break

            # else:
            loss_value = train_step_2(dataset, model, optimizer)

            # Log every 50 batches.
            if i % 50 == 0:
                print(
                    "Training loss, part 2 (for one batch) at step %d: %.4f"
                    % (i, float(loss_value))
                )
                training_loss_2.append(float(loss_value))
                print("Seen so far: %d samples" % ((i + 1) * hyperparameters["Batch Size"]))
                print("{}/{}".format((i + 1), len(train_gen_p2)))

        print("Time taken: %.2fs" % (time.time() - start_time))

    train_time_2 = time.time() - train_start_time_2

    return train_time_1, train_time_2, freeze_network_wts(model), seed_list, training_errors_1, validation_errors_1, \
           training_loss_1, training_loss_2


def testing(root_dir, model, fname="inaccurate_results", data=True):
    """
    Performs the testing phase for a given model (trained).
    Checks the accuracy and the speed (fps)
    :param root_dir: directory path that contains the csv files for testing dataset generation
    :param model: Trained model
    :return: Accuracy and Frame per Second (fps)
    """
    # Get generator for testing phase
    image_gen = ImageDataGenerator(rescale=1. / 255)
    test_df = pd.read_csv(os.path.join(root_dir, 'test_dataset_no_gates_only.csv'))
    test_gen = image_gen.flow_from_dataframe(dataframe=test_df,
                                             directory=None,
                                             x_col='Full_path',
                                             y_col=['distance_rho', 'distance_theta', 'distance_phi', 'yaw_diff'],
                                             target_size=(200, 300),
                                             color_mode='rgb',
                                             class_mode='raw',
                                             batch_size=1,
                                             shuffle=False)
    print('ImageDataGenerators for testing successful!')

    # Test the speed of the model
    fps = test_model_speed(model, test_gen)

    # Test the model
    acc, not_acc, crit_err = test_model(test_gen, model, fname=fname, data=data)  # pass it only a fraction of the new_train_gen
    accuracy = acc / (acc + not_acc) * 100
    print("Number of accurate predictions: {}".format(acc))
    print("Number of inaccurate predictions: {}".format(not_acc))
    print("Predictions accuracy percentage: {}".format(accuracy))
    print("There were {} cases where the algorithm was way off ".format(crit_err))
    print("which corresponds to {0:.2f}% of all cases".format(crit_err / (acc + not_acc) * 100))

    return fps, accuracy