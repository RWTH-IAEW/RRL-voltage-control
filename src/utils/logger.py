import wandb
import os
import torch

def load_model(path):
    model = torch.load(path,
                       map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


class Logger:

    def __init__(self, run_name, exp_name=None):
        self.run_name = run_name
        self.exp_name = exp_name

        self.exp_folder = "" if self.exp_name is None else self.exp_name
        self.run_folder = os.path.join(self.exp_folder, self.run_name)
        self.log_dir = self.make_logdir(self.run_folder)

    def init_wandb_logging(self, wandb_prj_name, config=None):
        """
        Sets up and initializes a Weights & Biases (wandb) logging session for the given notebook and run.
        Args:
            wandb_prj_name: string specifying the name of the wandb project to log to
            config: optional dictionary of experiment configuration values to log to wandb
        Returns:
            None
        """
        wandb.login()
        wandb.init(
            dir=self.log_dir,
            project=wandb_prj_name,
            name=self.run_name,
            config=config,
        )

    def get_folder_relative(self, folder_name, create=True):
        """
        Creates a folder with the given name in the current directory and returns the absolute path to the folder. The current directory is wrt the directory of the notebook that calls this function
        Args:
            folder_name: A string specifying the name of the folder to create.
            create: A boolean indicating whether to create the env if it does not exist.
        Returns:
            A tuple containing:
            - The absolute path to the newly created folder or existing folder.
            - A boolean indicating whether the folder already existed.
        """
        folder_path = os.path.join(self.log_dir, folder_name)
        if not os.path.isdir(folder_path):
            if not create:
                assert False, "Following folder does not exist %s" % (folder_path)
            os.makedirs(folder_path)
            folder_already_exist = False
        else:
            folder_already_exist = True
        return folder_path, folder_already_exist

    def make_logdir(self, run_folder):
        log_dir = os.path.join(os.path.abspath('logs'), run_folder)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, model, model_name='agent_model'):
        folder_path, _ = self.get_folder_relative('models')
        model_full_path = os.path.join(folder_path, f"{model_name}.pt")
        with open(model_full_path, 'wb') as f:
            torch.save(model, f)

    def load_model(self, model_name='agent_model'):
        model_full_path = os.path.join(self.log_dir, 'models', f"{model_name}.pt")
        model = torch.load(model_full_path,
                           map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return model
