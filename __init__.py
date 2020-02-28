from goalrepresent.config import Config
from goalrepresent.core import BaseEncoder, BaseModel, BaseEvaluationModel, BaseRepresentation
import goalrepresent.datasets
import goalrepresent.dnn
import goalrepresent.models
import goalrepresent.representations
import goalrepresent.evaluationmodels
import goalrepresent.gui



# version meaning: <major-release>.<non-compatible-update>.<compatible-update>
#  - major-release: Major new realeases
#  - non-compatible-update: Changes which do not allow to run previous experiments with the framework.
#  - compatible-update: Changes which allow to run previous experiments with the framework
__version__ = '0.8.0'
