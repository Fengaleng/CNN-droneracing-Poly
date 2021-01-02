import tensorflow as tf
import matplotlib.pyplot as plt


# Fonction pour geler les paramètres lors de la première étape du training
# layers[30], layers[32], layers[34] correspondent aux layers du MLP pour la covariance
def freeze_network_wts_p1(model):
    model.layers[42].trainable = False
    model.layers[44].trainable = False
    # model.layers[34].trainable = False
    print(model.summary())
    return model


def freeze_network_wts_p2(model):
    for layers in model.layers:
        layers.trainable = False
    model.layers[42].trainable = True
    model.layers[44].trainable = True
    # model.layers[34].trainable = True
    print(model.summary())
    return model


def unfreeze_network_wts(model):
    model.trainable = True
    print(model.summary())
    return model


def freeze_network_wts(model):
    model.trainable = False
    print(model.summary())
    return model


# custom loss function
def covariance_loss_function(prediction, ground_truth, covariance):
    """
    Takes in batch of predictions for the distance (rho, theta, phi, yaw)
    Then it compares them to their ground truth counterparts
    Finally, the cost is calculated with the covariance

    The cost is minimized through the following function which will give us the correct covariance
    """
    ground_truth = tf.dtypes.cast(ground_truth, tf.float64)
    covariance = tf.dtypes.cast(covariance, tf.float64)
    prediction = tf.dtypes.cast(prediction, tf.float64)
    tensor_form = tf.math.divide(
        tf.math.divide(tf.pow(tf.math.subtract(ground_truth, prediction), 2), covariance) + tf.math.log(covariance), 2)
    loss = tf.math.reduce_sum((tf.math.reduce_sum(tensor_form, axis=0)),
                              axis=0)  # ground truth and prediction must be the same dtype

    return loss


# Training step function v.2
@tf.function
def train_step_1(dataset, model, loss, optimizer, train_metric):
    # Training Parameters
    rho_wgt = 1
    theta_wgt = 0.5
    phi_wgt = 0.5
    yaw_wgt = 0.5
    x, y = dataset
    with tf.GradientTape() as tape:
        mean, _ = model(x, training=True)
        loss_value = loss(y, mean)
    grads_mean = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads_mean, model.trainable_weights))
    train_metric.update_state(y, mean)
    # sample_weight=[rho_wgt, theta_wgt, phi_wgt, yaw_wgt])

    return loss_value


@tf.function
def train_step_2(dataset, model, optimizer):
    x, y = dataset
    with tf.GradientTape() as tape:
        mean, cov = model(x, training=True)
        loss_value = covariance_loss_function(mean, y, cov)
    grads_mean = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads_mean, model.trainable_weights))

    return loss_value


@tf.function
def test_step(dataset, model, val_metric):
    # Validation metric weights
    rho_wgt = 1
    theta_wgt = 0.5
    phi_wgt = 0.5
    yaw_wgt = 0.5

    x, y = dataset
    mean, cov = model(x, training=False)
    val_metric.update_state(y, mean)
    # sample_weight=[rho_wgt, theta_wgt, phi_wgt, yaw_wgt])


def test_model(dataset, model, fname="inaccurate_results", data=True):
    """
    Test the model and print out random image samples with the prediction and ground-truth
    data (bool): if True write the difference between the prediction and teh ground truth to a file.
    """
    # Plot out random images with predicted labels and the correct ones
    acc = 0
    not_acc = 0
    crit_err = 0

    if data:
        # Write to a file the values that are not accurate
        not_acc_file = open(fname, "w")
        not_acc_file.write("Rho Diff \t Theta Diff \t Phi Diff \t Yaw Diff \n")

    for i, elem in enumerate(dataset):

        if i == len(dataset):
            break

        # print(image.shape) (1, 200, 300, 1)
        image = elem[0][0, :, :, 0]
        # print(image.shape) (200, 300)
        mean, cov = model(elem, training=False)

        # Calculate the difference of each returned value
        rho_diff = abs(elem[1][0][0] - mean[0, 0])
        theta_diff = abs(elem[1][0][1] - mean[0, 1])
        phi_diff = abs(elem[1][0][2] - mean[0, 2])
        yaw_diff = abs(elem[1][0][3] - mean[0, 3])

        # if rho_diff <= 0.1732 and theta_diff <= 0.1732 and phi_diff <= 0.1732 and yaw_diff <= 0.174:
        if rho_diff <= 0.25 and theta_diff <= 0.2 and phi_diff <= 0.2 and yaw_diff <= 0.2:
            # if rho_diff <= 1.0 and theta_diff <= 0.2 and phi_diff <= 0.2 and yaw_diff <= 0.2:
            acc += 1
            #             plt.figure(i + 1)
            #             plt.imshow(image)
            #             text = (
            #                 'Predictions-> Rho: {0:.3f}, Theta: {1:.3f}, Phi: {2:.3f}, Yaw: {3:.3f}'.format(mean[0, 0],
            #                                                                                                 mean[0, 1],
            #                                                                                                 mean[0, 2],
            #                                                                                                 mean[0, 3]),
            #                 'Covariance-> Rho: {0:.3f}, Theta: {1:.3f}, Phi: {2:.3f}, Yaw: {3:.3f}'.format(cov[0, 0],
            #                                                                                                cov[0, 1],
            #                                                                                                cov[0, 2],
            #                                                                                                cov[0, 3]),
            #                 'Ground Truth-> Rho: {0:.3f}, Theta: {1:.3f}, Phi: {2:.3f}, Yaw: {3:.3f}'.format(elem[1][0][0],
            #                                                                                                  elem[1][0][1],
            #                                                                                                  elem[1][0][2],
            #                                                                                                  elem[1][0][3]))
            #             plt.text(0, 0, text[0])
            #             plt.text(0, 100, text[1])
            #             plt.text(0, 190, text[2])
            #             plt.show()
            print('Accurate! {}/{}'.format(acc, i + 1))
        else:
            not_acc += 1

            if data:
                # Write the data to the file
                input1 = str(float(mean[0, 0]))
                input2 = str(float(mean[0, 1]))
                input3 = str(float(mean[0, 2]))
                input4 = str(float(mean[0, 3]))

                not_acc_file.write(input1)
                not_acc_file.write(" ; ")
                not_acc_file.write(input2)
                not_acc_file.write(" ; ")
                not_acc_file.write(input3)
                not_acc_file.write(" ; ")
                not_acc_file.write(input4)
                not_acc_file.write(" ; ")

                ## Only for the testing of images without gates
                # not_acc_file.write(str(int(25984 + i)))
                # not_acc_file.write(" ; ")
                # not_acc_file.write(str(round(float(rho_diff), 3)))
                # not_acc_file.write(" ; ")
                # not_acc_file.write(str(round(float(theta_diff), 3)))
                # not_acc_file.write(" ; ")
                # not_acc_file.write(str(round(float(phi_diff), 3)))
                # not_acc_file.write(" ; ")
                # not_acc_file.write(str(round(float(yaw_diff), 3)))
                # not_acc_file.write(" ; \n")

            # Show images with predictions of not accurate ones at every 100 incorrect predictions
            if not_acc % 50 == 0:
                plt.figure(i + 1)
                plt.imshow(image)
                text = (
                    'Predictions-> Rho: {0:.3f}, Theta: {1:.3f}, Phi: {2:.3f}, Yaw: {3:.3f}'.format(mean[0, 0],
                                                                                                    mean[0, 1],
                                                                                                    mean[0, 2],
                                                                                                    mean[0, 3]),
                    'Covariance-> Rho: {0:.3f}, Theta: {1:.3f}, Phi: {2:.3f}, Yaw: {3:.3f}'.format(cov[0, 0], cov[0, 1],
                                                                                                   cov[0, 2],
                                                                                                   cov[0, 3]),
                    'Ground Truth-> Rho: {0:.3f}, Theta: {1:.3f}, Phi: {2:.3f}, Yaw: {3:.3f}'.format(elem[1][0][0],
                                                                                                     elem[1][0][1],
                                                                                                     elem[1][0][2],
                                                                                                     elem[1][0][3]))
                plt.text(0, 0, text[0])
                plt.text(0, 100, text[1])
                plt.text(0, 190, text[2])
                plt.show()

            print('Not Accurate! {}/{}'.format(acc, i + 1))

            # Number of critical cases
            if rho_diff >= 0.5 or theta_diff >= 0.5 or phi_diff >= 0.5 or yaw_diff >= 0.5:
                crit_err += 1

    print("Testing complete!")
    not_acc_file.close()  # close file
    return acc, not_acc, crit_err
