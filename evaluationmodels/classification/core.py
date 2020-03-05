import bisect
import goalrepresent as gr
from goalrepresent.dnn.losses import losses
from goalrepresent.helper import datahelper
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid


class BaseClassifierModel(gr.BaseEvaluationModel):
    """
    Base Classifier Model Class
    Squeleton to follow for each classifier model, here simple ZeroR classifier
    """

    @staticmethod
    def default_config():
        default_config = gr.BaseEvaluationModel.default_config()

        default_config.output_name = "classification"

        # loss parameters
        default_config.loss = gr.Config()
        default_config.loss.name = "SVMClassifier"
        default_config.loss.parameters = gr.Config()

        # logging config (will save every X epoch)
        default_config.logging = gr.Config()
        default_config.logging.record_loss_every = 1
        default_config.logging.record_top10_images_every = 1

        # checkpoint config
        default_config.checkpoint = gr.Config()
        default_config.checkpoint.folder = None
        default_config.checkpoint.save_model_every = 10

        return default_config

    def __init__(self, representation_model, config=None, **kwargs):
        super().__init__(representation_model, config=config, **kwargs)

        # loss function
        self.set_loss(self.config.loss.name, self.config.loss.parameters)

        # n_epochs is incremented after each training iteration (in the case of SVM only 1 epoch)
        self.n_epochs = 0

    def set_loss(self, loss_name, loss_parameters):
        loss_class = losses.get_loss(loss_name)
        self.loss_f = loss_class(**loss_parameters)

        # update config
        self.config.loss.name = loss_name
        self.config.loss.parameters = gr.config.update_config(loss_parameters, self.config.loss.parameters)

    def forward(self, x):
        n_samples = x.size(0)
        predicted_y = torch.FloatTensor(n_samples, self.n_classes).fill_(0.0)
        predicted_y[:, self.majority_class] = 1.0
        return predicted_y

    def do_evaluation_pass(self, dataloader, logger=None):
        if isinstance(self, gr.dnn.BaseDNN):
            self.eval()
        predictions = {}
        losses = {}
        log_top10_images = (logger is not None) and (self.n_epochs % self.config.logging.record_top10_images_every == 0)
        log_loss = (logger is not None) and (self.n_epochs % self.config.logging.record_loss_every == 0)

        if log_top10_images:
            top10_scores = [sys.float_info.max] * 10
            top10_images = [None] * 10
            top10_titles = [None] * 10
            bottom10_scores = [0] * 10
            bottom10_images = [None] * 10
            bottom10_titles = [None] * 10

        # loop over the different batches
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                x = Variable(data["obs"])
                y = Variable(data["label"]).squeeze()
                if isinstance(self, gr.dnn.BaseDNN):
                    y = self.push_variable_to_device(y)
                # forward
                batch_predictions = self.forward(x)
                loss_inputs = {"predicted_y": batch_predictions, "y": y}
                batch_losses = self.loss_f(loss_inputs, reduction="none")

                # save results
                if 'predicted_y' in predictions:
                    predictions['predicted_y'] = np.vstack(
                        [predictions['predicted_y'], batch_predictions.detach().cpu().numpy()])
                else:
                    predictions['predicted_y'] = batch_predictions.detach().cpu().numpy()

                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = np.expand_dims(v.detach().cpu().numpy(), axis=-1)
                    else:
                        losses[k] = np.vstack([losses[k], np.expand_dims(v.detach().cpu().numpy(), axis=-1)])

                # update top-10 and bottom-10 classified images
                if log_top10_images:
                    for img_idx_wrt_batch, img in enumerate(x.cpu().data):
                        ce_loss = batch_losses["total"][img_idx_wrt_batch].data.item()
                        if ce_loss < top10_scores[-1]:
                            i = bisect.bisect(top10_scores, ce_loss)
                            top10_scores.insert(i, ce_loss)
                            top10_scores = top10_scores[:-1]
                            top10_images.insert(i, img.cpu().detach())
                            top10_images = top10_images[:-1]
                            pred_score, pred_y = batch_predictions[img_idx_wrt_batch].max(0)
                            title = "{} \n {:.2f}".format(int(pred_y), ce_loss)
                            top10_titles.insert(i, title)
                            top10_titles = top10_titles[:-1]
                        elif ce_loss > bottom10_scores[0]:
                            i = bisect.bisect(bottom10_scores, ce_loss)
                            bottom10_scores.insert(i, ce_loss)
                            bottom10_scores = bottom10_scores[1:]
                            bottom10_images.insert(i, img.cpu().detach())
                            bottom10_images = bottom10_images[1:]
                            pred_score, pred_y = batch_predictions[img_idx_wrt_batch].max(0)
                            title = "{} \n {:.2f}".format(int(pred_y), ce_loss)
                            bottom10_titles.insert(i, title)
                            bottom10_titles = bottom10_titles[1:]

                            # log results
        if log_top10_images:
            vizu_tensor_list = [None] * (2 * 10)
            vizu_tensor_list[0::2] = [top10_images[n] for n in range(10)]
            vizu_tensor_list[1::2] = [
                torch.from_numpy(datahelper.string2imgarray(top10_titles[n], top10_images[n].size())) for n in
                range(10)]
            img = make_grid(vizu_tensor_list, nrow=2, padding=0)
            logger.add_image("top10_classifications", img, self.n_epochs)
            vizu_tensor_list = [None] * (2 * 10)
            vizu_tensor_list[0::2] = [bottom10_images[n] for n in range(10)]
            vizu_tensor_list[1::2] = [
                torch.from_numpy(datahelper.string2imgarray(bottom10_titles[n], bottom10_images[n].size())) for n in
                range(10)]
            img = make_grid(vizu_tensor_list, nrow=2, padding=0)
            logger.add_image("bottom10_classifications", img, self.n_epochs)

        if log_loss:
            for k, v in losses.items():
                logger.add_scalars("loss/{}".format(k), {"valid": np.mean(v)}, self.n_epochs)

        return predictions, losses

    def save_predictions(self, predictions, npz_filepath):
        np.savez(npz_filepath, predictions=predictions)

    def visualize_results(self):
        pass

    def run_training(self, train_loader=None, valid_loader=None, keep_best_model=True, logger=None):
        pass

    def run_representation_testing(self, dataloader, testing_config=None):
        test_data = dict()

        # run testing on the test data
        test_predictions, test_losses = self.do_evaluation_pass(dataloader)

        test_data["predictions"] = test_predictions
        test_data["error"] = test_losses

        return test_data

    def update_hyperparameters(self, hyperparameters):
        """
        hyperparameters: dictionary of "name": value (value should be a float)
        """
        for hyperparam_key, hyperparam_val in hyperparameters.items():
            if hasattr(self, hyperparam_key):
                if isinstance(hyperparam_val, gr.Config):
                    setattr(self, hyperparam_key,
                            gr.config.update_config(hyperparam_val, getattr(self, hyperparam_key)))
                else:
                    setattr(self, hyperparam_key, hyperparam_val)

    def save_checkpoint(self, checkpoint_filepath):
        # save classifier if we want to relaunch testing from that point
        checkpoint = {
            "type": self.__class__.__name__,
            "init_params": self.init_params,
            "majority_class": self.majority_class,
        }

        torch.save(checkpoint, checkpoint_filepath)


def get_model(model_name):
    """
    model_name: string such that the model called is <model_name>Model
    """
    return eval("gr.evaluationmodels.{}Model".format(model_name))
