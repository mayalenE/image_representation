from abc import ABCMeta
from addict import Dict
from image_representation.representations.torch_nn import encoders
import torch
from torch import nn

EPS = 1e-12


class BDiscriminator(nn.Module, metaclass=ABCMeta):
    '''
    Base Discriminator class
    '''

    @staticmethod
    def default_config():
        default_config = Dict()

        default_config.n_latents = 10

        return default_config

    def __init__(self, config=None, **kwargs):
        nn.Module.__init__(self)
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)


def get_discriminator(model_architecture):
    '''
    model_architecture: string such that the class discriminator called is <model_architecture>Discriminator
    '''
    model_architecture = model_architecture.lower().capitalize()
    return eval("{}Discriminator".format(model_architecture))


""" ========================================================================================================================
Discriminator Modules 
========================================================================================================================="""


class BurgessDiscriminator(BDiscriminator):

    def __init__(self, config=None, **kwargs):
        encoder = encoders.BurgessEncoder(config=config, **kwargs)
        BDiscriminator.__init__(self, config=config, **kwargs)

        # inference x
        self.infer_x = nn.Sequential()
        self.infer_x.lf = encoder.lf
        self.infer_x.gf = encoder.gf

        hidden_dim_x = self.config.hidden_dim
        hidden_dim_joint = self.config.hidden_dim * 2
        self.infer_joint = nn.Sequential(
            nn.Linear(hidden_dim_x + self.config.n_latents, hidden_dim_joint),
            nn.ReLU(),
            nn.Linear(hidden_dim_joint, hidden_dim_joint),
            nn.ReLU(),
            nn.Linear(hidden_dim_joint, 1),
        )

    def forward(self, x, z):
        output_x = self.infer_x(x)
        output = self.infer_joint(torch.cat((output_x, z), 1))
        return output


class HjelmDiscriminator(BDiscriminator):

    def __init__(self, config=None, **kwargs):
        encoder = encoders.HjelmEncoder(config=config, **kwargs)
        BDiscriminator.__init__(self, config=config, **kwargs)

        # inference x
        self.infer_x = nn.Sequential()
        self.infer_x.lf = encoder.lf
        self.infer_x.gf = encoder.gf

        hidden_dim_x = self.config.hidden_dim
        hidden_dim_joint = self.config.hidden_dim * 2
        self.infer_joint = nn.Sequential(
            nn.Linear(hidden_dim_x + self.config.n_latents, hidden_dim_joint),
            nn.ReLU(),
            nn.Linear(hidden_dim_joint, hidden_dim_joint),
            nn.ReLU(),
            nn.Linear(hidden_dim_joint, 1),
        )

    def forward(self, x, z):
        if self.training and x.size(0) == 1:
            self.eval()
            output_x = self.infer_x(x)
            self.train()

        else:
            output_x = self.infer_x(x)
        output = self.infer_joint(torch.cat((output_x, z), 1))
        return output


class DumoulinDiscriminator(BDiscriminator):

    def __init__(self, config=None, with_dropout=True, **kwargs):
        encoder = encoders.DumoulinEncoder(config=config, **kwargs)
        BDiscriminator.__init__(self, config=config, **kwargs)

        # inference x
        self.infer_x = nn.Sequential()
        self.infer_x.lf = encoder.lf
        self.infer_x.gf = encoder.gf

        hidden_dim_x = 512
        hidden_dim_z = 1024
        hidden_dim_joint = 2048

        self.infer_z = nn.Sequential(
            nn.Conv2d(self.config.n_latents, hidden_dim_z, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim_z, hidden_dim_z, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
        )

        self.infer_joint = nn.Sequential(
            nn.Conv2d(hidden_dim_z + hidden_dim_x, hidden_dim_joint, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim_joint, hidden_dim_joint, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim_joint, 1, kernel_size=1, stride=1)
        )

        self.with_dropout = with_dropout
        if self.with_dropout:
            infer_x_lf_dropout_modules = []
            for module_name, module in self.infer_x.lf.named_children():
                module_list = [sub_mod for sub_mod in module.children()]
                module_list.insert(3, nn.Dropout2d(0.2))
                module_list.append(nn.Dropout2d(0.2))
                infer_x_lf_dropout_modules.append(nn.Sequential(*module_list))
            self.infer_x.lf = nn.Sequential(*infer_x_lf_dropout_modules)
            infer_x_gf_dropout_modules = []
            for module in self.infer_x.gf.children():
                module_list = [sub_mod for sub_mod in module.children()]
                module_list.insert(3, nn.Dropout2d(0.2))
                module_list.append(nn.Dropout2d(0.2))
                infer_x_gf_dropout_modules.append(nn.Sequential(*module_list))
            self.infer_x.gf = nn.Sequential(*infer_x_gf_dropout_modules)
            module_list = [mod for mod in self.infer_z.children()]
            module_list.insert(2, nn.Dropout2d(0.2))
            module_list.append(nn.Dropout2d(0.2))
            self.infer_z = nn.Sequential(*module_list)
            module_list = [mod for mod in self.infer_joint.children()]
            module_list.insert(2, nn.Dropout2d(0.2))
            module_list.insert(-1, nn.Dropout2d(0.2))
            self.infer_joint = nn.Sequential(*module_list)

    def forward(self, x, z):
        if len(z.size()) == 2:
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)
        if self.training and x.size(0) == 1:
            self.eval()
            output_x = self.infer_x(x)
            self.train()

        else:
            output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        output_joint = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        return output_joint
