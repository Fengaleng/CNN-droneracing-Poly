from data_manip import csv_dataset_generator
from model import resnet11, training,  load_model, testing
from show_result import save_model, plot_fig
import tensorflow as tf

# Data Preprocessing Section
root_dir = 'C:/Users/Feng Yang Chen/Documents/POLYTECHNIQUE/E2020 - Stage CRSNG - David Saussié/simulation_training_data/Training'
csv_dataset_generator(root_dir)

# Create the ResNet model
model = resnet11(300, 200, 3, 4)

# Hyperparameters for the training
hyperparam = {"Learning Rate": 0.001,
              "Beta 1": 0.9,
              "Beta 2": 0.999,
              "epsilon": 0.0000001,
              "Batch Size": 32,
              "Epoch 1": 50,
              "Epoch 2": 10
              }

# Metrics, loss function and optimizer declaration
train_metric = tf.keras.metrics.MeanAbsoluteError()
val_metric = tf.keras.metrics.MeanAbsoluteError()
mae = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparam["Learning Rate"],
                                     beta_1=hyperparam["Beta 1"],
                                     beta_2=hyperparam["Beta 2"],
                                     epsilon=hyperparam["epsilon"],
                                     amsgrad=True,
                                     name='Adam')

# Train the algorithm
train_time_1, train_time_2, model, seed_list, training_errors_1, validation_errors_1, \
training_loss_1, training_loss_2 = training(root_dir,
                                            hyperparam,
                                            model,
                                            train_metric,
                                            val_metric,
                                            mae,
                                            optimizer)

# Save the model with the relevant plots (update the path accordingly)
save_path = 'C:/Users/Feng Yang Chen/Documents/POLYTECHNIQUE/E2020 - Stage CRSNG - David Saussié/Script/Saved Models/2020-08-13-p2'
model_name = "Resnet11_model_"
save_model(save_path,
           model_name,
           model,
           hyperparam["Epoch 1"],
           train_time_1,
           hyperparam["Learning Rate"],
           hyperparam["Batch Size"],
           hyperparam["Beta 1"],
           hyperparam["Beta 2"],
           hyperparam["epsilon"],
           seed_list,
           hyperparam["Epoch 2"],
           train_time_2
           )
# Part 1 training plots
plot_fig(training_loss=training_loss_1,
         training_errors=training_errors_1,
         validation_errors=validation_errors_1,
         num_epoch=hyperparam["Epoch 1"],
         save_path=save_path,
         training_part=1)
# Part 2 training plots
plot_fig(training_loss=training_loss_2,
         num_epoch=hyperparam["Epoch 2"],
         save_path=save_path,
         training_part=2)

# Load the model
model = load_model(save_path, (model_name+".h5"))

# Test algorithm
fps, accuracy = testing(root_dir, model, data=False)

