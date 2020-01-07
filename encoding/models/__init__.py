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
from .pcnet import *
from .dasignet import *
from .ASGNet import *
from .glcnet5 import *
from .glcnet_topkpam import *
from .glcnet_aca import *
from .glcnet_amca import *
from .glcnet6 import *
from .danet_pam import *
from .new_psp3 import *
def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'danet': get_danet,
        'danet_pam': get_danet_pam,
        'msdanet': get_msdanet,
        'msdanet2': get_msdanet2,
        'mvdanet': get_mvdanet,
        'cascade_mvdanet': get_cascade_mvdanet,
        'mvdanet2': get_mvdanet2,
        'glcnet': get_glcnet,
        'pcnet': get_pcnet,
        'dasignet': get_dasignet,
        'asgnet': get_asgnet,
        'glcnet5': get_glcnet5,
        'glcnet6': get_glcnet6,
        'glcnet5_topkpam': get_glcnet5_topkpam,
        'glcnet5_aca': get_glcnet5_aca,
        'glcnet5_amca': get_glcnet5_amca,
        'new_psp3': get_new_psp3net,
    }
    return models[name.lower()](**kwargs)
