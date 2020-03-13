from goalrepresent.models.baselines.decomposition import PCAModel
from goalrepresent.models.baselines.manifold import TSNEModel, UMAPModel
from goalrepresent.models.dim import DIMModel
from goalrepresent.models.vaes import VAEModel, BetaVAEModel, AnnealedVAEModel, BetaTCVAEModel
from goalrepresent.models.gans import BiGANModel, VAEGANModel
from goalrepresent.models.progressivetree import ProgressiveTreeModel
from goalrepresent.models.quadruplet import VAEQuadrupletModel, BetaVAEQuadrupletModel, AnnealedVAEQuadrupletModel, \
    BetaTCVAEQuadrupletModel, BiGANQuadrupletModel, VAEGANQuadrupletModel
from goalrepresent.models.triplet import VAETripletModel, BetaVAETripletModel, AnnealedVAETripletModel, \
    BetaTCVAETripletModel, BiGANTripletModel, VAEGANTripletModel
