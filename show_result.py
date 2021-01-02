import os
import shutil
import matplotlib.pyplot as plt
import numpy as np


def save_model(path, model_name, model, num_epoch_1, time_1, learning_rate, batch_size, beta_1, beta_2, epsilon, seed,
               num_epoch_2=None, time_2=None):
    # Create a new folder for the model
    if not os.path.isdir(path):
        os.makedirs(path)

    # Create a txt file with hyperparameters in it
    file = open(model_name + '_hyperparam.txt', "a")
    file.write(model_name + ' trained in-> part 1 ' + str(time_1) + ' seconds\n\n')
    file.write('Hyperparameters used \n')
    file.write('Learning Rate: ' + str(learning_rate) +
               '\tBatch Size: ' + str(batch_size) +
               '\t# of epochs in training-> part 1: ' + str(num_epoch_1) +
               '\nAdam Optimizer-> beta 1: ' + str(beta_1) +
               '\tbeta 2: ' + str(beta_2) +
               '\tepsilon: ' + str(epsilon))
    file.write('Seed: ')
    for i in range(len(seed)):
        file.write(str(seed[i]) + ' ')

    if num_epoch_2 is not None and time_2 is not None:
        file.write('\nTraining part 2: \n' +
                   '# of epochs: ' + str(num_epoch_2) +
                   '\ttrained in: ' + str(time_2) + ' seconds')
    file.close()

    # Move the file to the correct location
    shutil.move(model_name + '_hyperparam.txt', path)
    print('Sucessfully Saved and Moved!')

    # Save the whole model in that directory
    model_name = model_name + '.h5'
    model.save(os.path.join(path, model_name))


def plot_fig(training_loss=None, training_errors=None, validation_errors=None, num_epoch=None, save_path=None,
             training_part=None):
    """
    Inputs:
    Lists with the relevant information to be displayed from the training and validation.
    Requires also the number of epochs
    """
    # Training loss
    if training_loss is not None:
        num_curves = num_epoch
        plt.figure(1)
        fig1 = plt.gcf()
        plt.xlabel('Steps in 1 epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        x = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
        plot_list = []
        label_list = []
        for i in range(int(num_curves)):
            label = 'Epoch {}'.format(i)
            train_loss, = plt.plot(x, training_loss[i * len(x):((i + 1) * len(x))], label=label)
            plot_list.append(train_loss)
            label_list.append(label)
        plt.legend(plot_list, label_list, loc='upper right')
        plt.show()
        if training_part == 1:
            fig1.savefig(os.path.join(save_path, 'Training_Loss_1.png'), dpi=100)
        else:
            fig1.savefig(os.path.join(save_path, 'Training_Loss_2.png'), dpi=100)

    # Training and Validation errors graph
    if training_errors is not None and validation_errors is not None:
        plt.figure(2)
        fig2 = plt.gcf()
        plt.xlabel('# epoch')
        plt.ylabel('Error')
        plt.title('Training Error Evolution')
        x = np.arange(num_epoch).tolist()
        train, = plt.plot(x, training_errors, label='Training Error')
        val, = plt.plot(x, validation_errors, label='Validation Error')
        plt.legend([train, val], ['Training Error', 'Validation Error'], loc='upper right')
        plt.show()
        fig2.savefig(os.path.join(save_path, 'Train_Val_Errors.png'), dpi=100)
        if training_part == 2:
            print('Careful, this is the second training part.\n'
                  'There should not be training errors nor validation errors.')
    print('Graph created and saved!')
