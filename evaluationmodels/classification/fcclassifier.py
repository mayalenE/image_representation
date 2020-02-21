import goalrepresent as gr
from goalrepresent import dnn, evaluationmodels
from goalrepresent.dnn.solvers import initialization
# import io
# import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
from torch import nn
from torch.autograd import Variable
# from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
import warnings

transform_to_PIL_image = ToPILImage()
transform_to_tensor = ToTensor()


class FCClassifierModel(dnn.BaseDNN, evaluationmodels.BaseClassifierModel):
    """
    FC Classifier Evaluation Model Class
    """

    @staticmethod
    def default_config():
        default_config = evaluationmodels.BaseClassifierModel.default_config()

        # freeze or not the plugged encoder
        default_config.freeze_encoder = True

        # network parameters
        default_config.network = gr.Config()
        default_config.network.name = "Linear"
        default_config.network.parameters = gr.Config()
        default_config.network.parameters.n_hidden_units = 128
        default_config.network.parameters.n_classes = 10

        # initialization parameters
        default_config.network.initialization = gr.Config()
        default_config.network.initialization.name = "pytorch"
        default_config.network.initialization.parameters = gr.Config()

        # device parameters
        default_config.device = gr.Config()
        default_config.device.use_gpu = True

        # loss parameters
        default_config.loss.name = "FCClassifier"

        # optimizer parameters
        default_config.optimizer = gr.Config()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = gr.Config()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5

        return default_config

    def __init__(self, representation_model, config=None, **kwargs):
        super(dnn.BaseDNN, self).__init__()  # calls nn.Module constructor
        super(nn.Module, self).__init__(representation_model, config=config,
                                        **kwargs)  # calls evaluationmodels.BaseClassifierModel constructor

        # define the device to use (gpu or cpu)
        if self.config.device.use_gpu and not torch.cuda.is_available():
            self.config.device.use_gpu = False
            warnings.warn("Cannot set model device as GPU because not available, setting it to CPU")

        # network FC
        self.set_network(self.config.network.name, self.config.network.parameters)
        ## freeze or not the pretrained encoder
        if self.config.freeze_encoder:
            for param in self.network.encoder.parameters():
                param.requires_grad = False
        else:
            self.network.encoder.train()
            for param in self.network.encoder.parameters():
                param.requires_grad = True
        self.init_network(self.config.network.initialization.name, self.config.network.initialization.parameters)
        self.set_device(self.config.device.use_gpu)

        # optimizer
        self.set_optimizer(self.config.optimizer.name, self.config.optimizer.parameters)

        self.n_epochs = 0

    def set_network(self, network_name, network_parameters):
        self.network = nn.Module()
        self.network.encoder = self.representation_encoder
        self.network.fc = nn.Sequential(
            nn.Linear(self.network.encoder.n_latents, network_parameters.n_hidden_units),
            nn.ReLU(),
            nn.Linear(network_parameters.n_hidden_units, network_parameters.n_classes)
        )

        # update config
        self.config.network.name = network_name
        self.config.network.parameters = gr.config.update_config(network_parameters, self.config.network.parameters)

    def init_network(self, initialization_name, initialization_parameters):
        initialization_class = initialization.get_initialization(initialization_name)
        # only initialize the classifier (keep the pretrained encoder)
        self.network.fc.apply(initialization_class)

        # update config
        self.config.network.initialization.name = initialization_name
        self.config.network.initialization.parameters = gr.config.update_config(initialization_parameters,
                                                                                self.config.network.initialization.parameters)

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        # if encoder frozen only optimize the classifier, else optimize alltogether
        if self.config.freeze_encoder:
            self.optimizer = optimizer_class(self.network.fc.parameters(), **optimizer_parameters)
        else:
            self.optimizer = optimizer_class(self.network.parameters(), **optimizer_parameters)

        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters,
                                                                   self.config.optimizer.parameters)

    def forward(self, x):
        x = self.push_variable_to_device(x)
        encoder_outputs = self.representation_encoder(x)
        return self.network.fc(encoder_outputs["z"])

    def train_epoch(self, train_loader, logger=None):
        self.train()
        losses = {}
        for data in train_loader:
            x = Variable(data["obs"])
            y = Variable(data["label"]).squeeze()
            y = self.push_variable_to_device(y)
            # forward
            batch_predictions = self.forward(x)
            loss_inputs = {"predicted_y": batch_predictions, "y": y}
            batch_losses = self.loss_f(loss_inputs, reduction=True)
            # backward
            loss = batch_losses["total"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # save losses
            for k, v in batch_losses.items():
                if k not in losses:
                    losses[k] = [v.data.item()]
                else:
                    losses[k].append(v.data.item())

        for k, v in losses.items():
            losses[k] = np.mean(v)

        self.n_epochs += 1

        return losses

    def run_training(self, train_loader=None, valid_loader=None, keep_best_model=True, logger=None):
        """
        logger: tensorboard X summary writer
        """
        # all the evaluation models are plugged to the frozen representation and trained for 100 epochs
        n_epochs = 50

        # Save the graph in the logger
        if logger is not None:
            dummy_input = torch.FloatTensor(1, self.network.encoder.n_channels, self.network.encoder.input_size[0],
                                            self.network.encoder.input_size[1]).uniform_(0, 1)
            dummy_input = self.push_variable_to_device(dummy_input)
            # logger.add_graph(nn.Sequential(representation.model.encoder, self), dummy_input)

        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        for epoch in range(n_epochs):
            t0 = time.time()
            train_losses = self.train_epoch(train_loader, logger=logger)
            t1 = time.time()

            if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                for k, v in train_losses.items():
                    logger.add_scalars("loss/{}".format(k), {"train": v}, self.n_epochs)
                logger.add_text("time/train", "Train Epoch {}: {:.3f} secs".format(self.n_epochs, t1 - t0),
                                self.n_epochs)

            if self.n_epochs % self.config.checkpoint.save_model_every == 0:
                self.save_checkpoint(os.path.join(self.config.checkpoint.folder, "current_weight_model.pth"))

            if do_validation:
                _, valid_losses = self.do_evaluation_pass(valid_loader, logger=logger)
                valid_loss = np.mean(valid_losses["total"])
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.save_checkpoint(os.path.join(self.config.checkpoint.folder, "best_weight_model.pth"))

        # if we want the representation to keep the model that performed best on the valid dataset
        if keep_best_model:
            best_model_path = os.path.join(self.config.checkpoint.folder, "best_weight_model.pth")
            if os.path.exists(best_model_path):
                best_model = torch.load(best_model_path, map_location="cpu")
                self.network.load_state_dict(best_model["network_state_dict"])
                self.optimizer.load_state_dict(best_model["optimizer_state_dict"])

    def run_representation_testing(self, dataloader, testing_config=None):
        test_data = dict()

        # run testing on the test data
        test_predictions, test_losses = self.do_evaluation_pass(dataloader)

        #  transform x -> probabilities predictions
        exp_predictions = np.exp(test_predictions["predicted_y"])
        test_predictions["predicted_y"] = exp_predictions / np.expand_dims(np.sum(exp_predictions, axis=1), axis=1)

        test_data["predictions"] = test_predictions
        test_data["error"] = test_losses

        return test_data
