import os
import pandas as pd


def csv_dataset_generator(root_dir):
    """
    Takes the root directory with all the images to generate the appropriate csv files for dataset generation later on.
    :param root_dir:
    :return csv files created:
    """
    list_of_runs = os.listdir(root_dir)

    # list that will contain the dataframes
    df_list = []
    df_no_gate_list = []
    df_no_gate_only_list = []

    num_run = 0
    for run in list_of_runs:
        # Do it for every run in the Training folder
        mypath = os.listdir(os.path.join(root_dir, run, 'images'))
        print(30 * '-')
        print('Starting: ' + run + ' ; {}/{}'.format(num_run + 1, len(list_of_runs)))

        for gate_folder in mypath:
            # Do it for every gate in every run
            print('Starting: ' + gate_folder)

            # Extract the labels from the dataframe
            # cols = ['distance_rho', 'distance_theta', 'distance_phi', 'yaw_diff', 'Image_name']
            df = pd.read_csv(os.path.join(root_dir, run, 'images', gate_folder, 'labels.csv'))  # , names=cols

            df_list.append(df)
            df_no_gate_list.append(df)

            # Specific case with no gates folders
            if os.path.isdir(os.path.join(root_dir, run, 'images', gate_folder, 'Sans Cible')):
                # cols = ['distance_rho', 'distance_theta', 'distance_phi', 'yaw_diff', 'Image_name']
                df_no_gate = pd.read_csv(
                    os.path.join(root_dir, run, 'images', gate_folder, 'Sans Cible', 'labels.csv'))  # , names=cols
                df_no_gate_list.append(df_no_gate)
                df_no_gate_only_list.append(df_no_gate)

        num_run += 1
        print(30 * '-')

    # Combine the dataframes together
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df_with_no_gate = pd.concat(df_no_gate_list, ignore_index=True)
    combined_df_with_no_gate_only = pd.concat(df_no_gate_only_list, ignore_index=True)
    combined_df.drop(['Image_name', 'Unnamed: 0'], axis=1, inplace=True)
    combined_df_with_no_gate.drop(['Image_name', 'Unnamed: 0'], axis=1, inplace=True)
    combined_df_with_no_gate_only.drop(['Image_name', 'Unnamed: 0'], axis=1, inplace=True)

    # Split datasets for training, validation and testing
    train_df, test_df = combined_df.loc[0:25983, :], combined_df.loc[25984:, :]

    # Convert to csv files (part 1 training and part 2 training)
    train_df.to_csv(os.path.join(root_dir, 'new_train_dataset_v2.csv'))
    test_df.to_csv(os.path.join(root_dir, 'test_dataset.csv'))
    combined_df_with_no_gate.to_csv(os.path.join(root_dir, 'full_dataset_with_no_gate.csv'))
    combined_df_with_no_gate_only.to_csv(os.path.join(root_dir, 'test_dataset_no_gates_only.csv'))

