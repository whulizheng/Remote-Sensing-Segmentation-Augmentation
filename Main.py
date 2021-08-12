from utilities import Constants, Utils
import os, shutil
import tensorflow as tf


def split_train_test(train_path, test_path):
    # Load the train set and the test set, respectively.
    train_set = tf.data.Dataset.list_files(train_path + '*.png')
    train_set = train_set.map(lambda x: Utils.load_image_train(x, Constants.SHAPE))
    train_set = train_set.shuffle(Constants.BUFFER_SIZE)
    train_set = train_set.batch(Constants.BATCH_SIZE)

    test_set = tf.data.Dataset.list_files(test_path + '*.png')
    test_set = test_set.map(lambda x: Utils.load_image_train(x, Constants.SHAPE))
    test_set = test_set.batch(Constants.BATCH_SIZE)

    return train_set, test_set


def model_training_loop():
    for dataset in target_datasets:
        train_path = 'datasets/' + dataset + '/TrainSet/'
        test_path = 'datasets/' + dataset + '/TestSet/'
        train_set, test_set = split_train_test(train_path=train_path, test_path=test_path)

        for model_name in target_models:
            for iteration in Constants.ITERATIONS:
                print(model_name + ' training on ' + dataset + ' at iteration ' + str(iteration) + ' starts......')
                my_output_dir = os.getcwd() + '/results/' + dataset + '/' + model_name + '/' + str(iteration)
                output_dir = Utils.create_directory(my_output_dir)
                # If the target directory exists, remove it and recreate it
                if output_dir is None:
                    shutil.rmtree(my_output_dir)
                    output_dir = Utils.create_directory(my_output_dir)
                model = Utils.create_model(model_name, output_dir)
                model.fit(train_set, test_set, Constants.EPOCHS)
                # Creation of the directory 'DONE' means all experiments are successfully conducted.
                Utils.create_directory(output_dir + '/DONE')
                print(model_name + ' training on ' + dataset + ' at iteration ' + str(iteration) + ' ended!')


def model_testing_loop():
    print()


if __name__ == '__main__':
    if Constants.USE_CPU:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    command_string = 'train_models'
    # command_string = 'test_models'
    # command_string = 'aggregate_results'
    # command_string = 'visualize_results'

    target_datasets = Constants.DATASETS[:]
    target_models = Constants.MODELS[0:1]

    if command_string == 'train_models':
        model_training_loop()
    elif command_string == 'test_models':
        model_testing_loop()
    elif command_string == 'aggregate_results':
        print('Not implemented yet!')
    elif command_string == 'visualize_results':
        print('Not implemented yet!')
    else:
        raise ValueError("No such command!")