import sys
import tensorflow as tf
from functools import partial
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from keras.layers import Input, Dense, Flatten
from stochnet.utils.file_organization import ProjectFileExplorer
from stochnet.utils.utils import get_train_and_validation_generator_w_scaler
from stochnet.classes.TopLayers import MixtureOutputLayer, MultivariateNormalDiagOutputLayer
from stochnet.applicative.histogram_w_gillespy import compute_histogram_distance, get_SSA_hist, get_SSA_traj
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, binom


def get_NN(nb_past_timesteps, nb_features, nb_hidden_nodes_1, max_norm_1,
           two_layers, nb_hidden_nodes_2, nb_components):
    input_tensor = Input(shape=(nb_past_timesteps, nb_features))
    flatten1 = Flatten()(input_tensor)

    NN_body = Dense(nb_hidden_nodes_1, kernel_constraint=maxnorm(max_norm_1),
                    activation='relu')(flatten1)
    if two_layers == 2:
        NN_body = Dense(nb_hidden_nodes_2, kernel_constraint=maxnorm(max_norm_1),
                        activation='relu')(NN_body)

    number_of_components = nb_components
    components = []
    for j in range(number_of_components):
        components.append(MultivariateNormalDiagOutputLayer(nb_features))

    TopModel_obj = MixtureOutputLayer(components)
    NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)
    return NN


def get_best_NN(param_list, get_model, scaler, train_gen, val_gen, callbacks):
    best_params = get_best_params(param_list, get_model, scaler, train_gen, val_gen)
    print("Parametri migliori: {0}.\n\n".format(best_params))
    print(param_list)
    best_NN = get_trained_NN(best_params, get_model, scaler, train_gen, val_gen, callbacks)
    return best_NN


def get_best_params(param_list, get_model, scaler, train_gen, val_gen):
    scores = []
    for params in param_list:
        callbacks = [EarlyStopping(monitor='val_loss', patience=3,
                                   verbose=1, mode='min')]
        NN = get_trained_NN(params, get_model, scaler, train_gen, val_gen, callbacks)
        score = scorer(NN=NN)
        scores.append(score)
        del NN

    best_index = scores.index(min(scores))
    best_params = param_list[best_index]
    return best_params


def get_trained_NN(params, get_model, scaler, train_gen, val_gen, callbacks):
    NN = get_model(**params)
    NN.memorize_scaler(scaler)
    result = NN.fit_generator(training_generator=train_gen,
                              samples_per_epoch=3 * 10**4, epochs=10, verbose=1,
                              callbacks=callbacks,
                              validation_generator=val_gen,
                              nb_val_samples=10**3)
    lowest_val_loss = min(result.history['val_loss'])
    print(lowest_val_loss)
    return NN


if __name__ == '__main__':
    sess = tf.Session()

    timestep = float(sys.argv[1])
    nb_past_timesteps = int(sys.argv[2])
    train_dataset_id = int(sys.argv[3])
    val_dataset_id = int(sys.argv[4])
    project_folder = str(sys.argv[5])
    model_id = int(sys.argv[6])

    project_explorer = ProjectFileExplorer(project_folder)

    train_explorer = project_explorer.get_DatasetFileExplorer(timestep, train_dataset_id)
    val_explorer = project_explorer.get_DatasetFileExplorer(timestep, train_dataset_id)

    train_gen, val_gen, scaler = get_train_and_validation_generator_w_scaler(train_explorer,
                                                                             val_explorer)

    nb_features = scaler.scale_.shape[0]

    param_grid = {'max_norm_1': randint(1, 5),
                  'nb_hidden_nodes_1': binom(500, 0.5),
                  'two_layers': randint(0, 1),
                  'nb_hidden_nodes_2': binom(120, 0.5)
                  }
    param_list = list(ParameterSampler(param_grid, n_iter=5))

    model_explorer = project_explorer.get_ModelFileExplorer(timestep, model_id)

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   verbose=1,
                                   mode='min'))

    checkpoint_filepath = model_explorer.weights_fp
    callbacks.append(ModelCheckpoint(checkpoint_filepath, monitor='val_loss',
                                     verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='min'))

    SSA_traj = get_SSA_traj(val_explorer)
    S_SSA_hist = get_SSA_hist(SSA_traj)
    nb_traj = SSA_traj.shape[1]
    scorer = partial(compute_histogram_distance, dataset_explorer=val_explorer,
                     S_SSA_hist=S_SSA_hist, nb_traj=nb_traj, sess=sess,
                     model_id=model_id, plot=False, log_results=False)
    get_model = partial(get_NN, nb_past_timesteps=nb_past_timesteps,
                        nb_features=nb_features, nb_components=2)
    NN = get_best_NN(param_list, get_model, scaler, train_gen, val_gen, callbacks)
    NN.load_weights(checkpoint_filepath)
    NN.save_model(model_explorer.keras_fp)
    NN.save(model_explorer.StochNet_fp)
