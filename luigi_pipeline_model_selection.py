#!/home/lucap/anaconda3/bin/python
import os
import luigi
import luigi.contrib.external_program
from importlib import import_module
from stochnet.utils.file_organization import ProjectFileExplorer


class GenerateDataset(luigi.contrib.external_program.ExternalPythonProgramTask):

    project_folder = luigi.Parameter(default='/home/lucap/Documenti/Tesi Magistrale/SIRS')
    dataset_id = luigi.IntParameter(default=1)
    nb_of_settings = luigi.IntParameter(default=25)
    nb_of_trajectories = luigi.IntParameter(default=100)
    timestep = luigi.FloatParameter(default=2**(-1))
    endtime = luigi.IntParameter(default=5)
    CRN_name = luigi.Parameter(default='SIRS')

    virtualenv = '/home/lucap/anaconda3/envs/py2'

    def program_args(self):
        program_module = import_module("stochnet.dataset.simulation_w_gillespy")
        program_address = program_module.__file__
        return ['python', program_address, self.dataset_id, self.nb_of_settings,
                self.nb_of_trajectories, self.timestep, self.endtime,
                self.project_folder, self.CRN_name]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_DatasetFileExplorer(self.timestep, self.dataset_id)
        return luigi.LocalTarget(dataset_explorer.dataset_fp)


class FormatDataset(luigi.contrib.external_program.ExternalPythonProgramTask):

    project_folder = luigi.Parameter(default='/home/lucap/Documenti/Tesi Magistrale/SIRS')
    dataset_id = luigi.IntParameter(default=1)
    timestep = luigi.FloatParameter(default=2**(-1))
    nb_past_timesteps = luigi.IntParameter(default=1)

    def requires(self):
        return GenerateDataset(project_folder=self.project_folder,
                               dataset_id=self.dataset_id,
                               timestep=self.timestep)

    def program_args(self):
        program_module = import_module("stochnet.utils.format_np_for_ML")
        program_address = program_module.__file__
        return ['python', program_address, self.nb_past_timesteps,
                self.dataset_id, self.timestep, self.project_folder]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_DatasetFileExplorer(self.timestep, self.dataset_id)
        return [luigi.LocalTarget(dataset_explorer.x_fp),
                luigi.LocalTarget(dataset_explorer.y_fp),
                luigi.LocalTarget(dataset_explorer.rescaled_x_fp),
                luigi.LocalTarget(dataset_explorer.rescaled_y_fp),
                luigi.LocalTarget(dataset_explorer.scaler_fp)]


class SelectAndTrainNN(luigi.contrib.external_program.ExternalPythonProgramTask):

    project_folder = luigi.Parameter(default='/home/lucap/Documenti/Tesi Magistrale/SIRS')
    training_dataset_id = luigi.IntParameter(default=1)
    validation_dataset_id = luigi.IntParameter(default=2)
    timestep = luigi.FloatParameter(default=2**(-1))
    nb_past_timesteps = luigi.IntParameter(default=1)
    model_id = luigi.IntParameter(default=1)
    nb_histogram_settings = luigi.IntParameter(default=20)

    def requires(self):
        return [FormatDataset(project_folder=self.project_folder,
                              dataset_id=self.training_dataset_id,
                              timestep=self.timestep,
                              nb_past_timesteps=self.nb_past_timesteps),
                FormatDataset(project_folder=self.project_folder,
                              dataset_id=self.validation_dataset_id,
                              timestep=self.timestep,
                              nb_past_timesteps=self.nb_past_timesteps),
                GenerateHistogramData(project_folder=self.project_folder,
                                      dataset_id=self.validation_dataset_id,
                                      timestep=self.timestep,
                                      nb_past_timesteps=self.nb_past_timesteps,
                                      nb_histogram_settings=self.nb_histogram_settings)]

    def program_args(self):
        program_address = os.path.join(self.project_folder,
                                       'model_selection_w_sklearn.py')
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.training_dataset_id,
                self.validation_dataset_id, self.project_folder,
                self.model_id]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        model_explorer = project_explorer.get_ModelFileExplorer(self.timestep, self.model_id)
        return [luigi.LocalTarget(model_explorer.weights_fp),
                luigi.LocalTarget(model_explorer.keras_fp),
                luigi.LocalTarget(model_explorer.StochNet_fp)]


class GenerateHistogramData(luigi.contrib.external_program.ExternalPythonProgramTask):

    project_folder = luigi.Parameter(default='/home/lucap/Documenti/Tesi Magistrale/SIRS')
    dataset_id = luigi.IntParameter(default=1)
    timestep = luigi.FloatParameter(default=2**(-1))
    nb_past_timesteps = luigi.IntParameter(default=1)
    nb_histogram_settings = luigi.IntParameter(default=15)
    nb_trajectories = luigi.IntParameter(default=500)
    CRN_name = luigi.Parameter(default='SIRS')

    virtualenv = '/home/lucap/anaconda3/envs/py2'

    def requires(self):
        return FormatDataset(project_folder=self.project_folder,
                             dataset_id=self.dataset_id,
                             timestep=self.timestep,
                             nb_past_timesteps=self.nb_past_timesteps)

    def program_args(self):
        program_module = import_module("stochnet.dataset.generator_for_histogram_w_gillespy")
        program_address = program_module.__file__
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.dataset_id,
                self.nb_histogram_settings, self.nb_trajectories,
                self.project_folder, self.CRN_name]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_DatasetFileExplorer(self.timestep, self.dataset_id)
        return [luigi.LocalTarget(dataset_explorer.histogram_settings_fp),
                luigi.LocalTarget(dataset_explorer.histogram_dataset_fp)]


class HistogramDistance(luigi.contrib.external_program.ExternalPythonProgramTask):

    project_folder = luigi.Parameter(default='/home/lucap/Documenti/Tesi Magistrale/SIRS')
    training_dataset_id = luigi.IntParameter(default=1)
    validation_dataset_id = luigi.IntParameter(default=2)
    test_dataset_id = luigi.IntParameter(default=3)
    timestep = luigi.FloatParameter(default=2**(-1))
    nb_past_timesteps = luigi.IntParameter(default=1)
    model_id = luigi.IntParameter(default=1)
    nb_histogram_settings = luigi.IntParameter(default=30)
    nb_trajectories = luigi.IntParameter(default=1000)

    def requires(self):
        return [SelectAndTrainNN(project_folder=self.project_folder,
                                 training_dataset_id=self.training_dataset_id,
                                 validation_dataset_id=self.validation_dataset_id,
                                 nb_histogram_settings=self.nb_histogram_settings,
                                 timestep=self.timestep,
                                 nb_past_timesteps=self.nb_past_timesteps,
                                 model_id=self.model_id),
                GenerateHistogramData(project_folder=self.project_folder,
                                      dataset_id=self.training_dataset_id,
                                      timestep=self.timestep,
                                      nb_past_timesteps=self.nb_past_timesteps,
                                      nb_histogram_settings=self.nb_histogram_settings),
                GenerateHistogramData(project_folder=self.project_folder,
                                      dataset_id=self.test_dataset_id,
                                      timestep=self.timestep,
                                      nb_past_timesteps=self.nb_past_timesteps,
                                      nb_histogram_settings=self.nb_histogram_settings)]

    def program_args(self):
        program_module = import_module("stochnet.applicative.histogram_w_gillespy")
        program_address = program_module.__file__
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.training_dataset_id,
                self.test_dataset_id, self.model_id,
                self.project_folder]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        train_explorer = project_explorer.get_DatasetFileExplorer(self.timestep, self.training_dataset_id)
        train_histogram_explorer = train_explorer.get_HistogramFileExplorer(self.model_id)
        test_explorer = project_explorer.get_DatasetFileExplorer(self.timestep, self.test_dataset_id)
        test_histogram_explorer = test_explorer.get_HistogramFileExplorer(self.model_id)
        return [luigi.LocalTarget(train_histogram_explorer.log_fp),
                luigi.LocalTarget(test_histogram_explorer.log_fp)]


if __name__ == '__main__':
    luigi.run(main_task_cls=HistogramDistance)
