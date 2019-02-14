from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .danet import *
from .msdanet import *
from .msdanet2 import *
from .mview_danet import *
from .mview_danet2 import *
from .cascade_mvdanet import *
from .GLCnet import *
from .glcnet_fast import *
from .glnet import *
from .pcnet import *
def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'danet': get_danet,
        'msdanet': get_msdanet,
        'msdanet2': get_msdanet2,
        'mvdanet': get_mvdanet,
        'cascade_mvdanet': get_cascade_mvdanet,
        'mvdanet2': get_mvdanet2,
        'glcnet': get_glcnet,
        'glcnet_fast': get_glcnet_fast,
        'glnet': get_glnet,
        'pcnet': get_pcnet,
    }
    return models[name.lower()](**kwargs)
