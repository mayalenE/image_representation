# main
from image_representation.representations.torch_nn.torch_nn_representation import TorchNNRepresentation

# utils
from image_representation.representations.torch_nn.losses import get_loss
from image_representation.representations.torch_nn.encoders import get_encoder
from image_representation.representations.torch_nn.decoders import get_decoder
from image_representation.representations.torch_nn.discriminators import get_discriminator

# image representation NN models
from image_representation.representations.torch_nn.dim import DIM
from image_representation.representations.torch_nn.vae import VAE, BetaVAE, AnnealedVAE, BetaTCVAE
from image_representation.representations.torch_nn.gan import BiGAN, VAEGAN
from image_representation.representations.torch_nn.clr import SimCLR, TripletCLR
from image_representation.representations.torch_nn.holmes_vae import HOLMES_VAE
from image_representation.representations.torch_nn.holmes_clr import HOLMES_CLR
from image_representation.representations.torch_nn.quadruplet import VAEQuadruplet, BetaVAEQuadruplet, AnnealedVAEQuadruplet, BetaTCVAEQuadruplet
from image_representation.representations.torch_nn.triplet import VAETriplet, BetaVAETriplet, AnnealedVAETriplet, BetaTCVAETriplet

__all__ = ["get_loss", "get_encoder", "get_decoder", "get_discriminator",
           "TorchNNRepresentation",
           "DIM",
           "VAE", "BetaVAE", "AnnealedVAE", "BetaTCVAE",
           "BiGAN", "VAEGAN",
           "SimCLR", "TripletCLR",
           "HOLMES_VAE", "HOLMES_CLR",
           "VAEQuadruplet", "BetaVAEQuadruplet", "AnnealedVAEQuadruplet", "BetaTCVAEQuadruplet",
           "VAETriplet", "BetaVAETriplet", "AnnealedVAETriplet", "BetaTCVAETriplet"]
