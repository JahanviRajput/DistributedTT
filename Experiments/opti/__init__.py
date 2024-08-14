from .opti import Opti
from .opti_nb import OptiNB
from .opti_opo import OptiOPO
from .opti_optimatt import OptiOptimatt
from .opti_portfolio import OptiPortfolio
from .opti_pso import OptiPSO
from .opti_protes import OptiProtes
from .opti_spsa import OptiSPSA
from .opti_ttopt import OptiTTOpt
from .opti_dips import OptiDiPTS
from .opti_noisy import OptiNoisyPTS
from .opti_protes_noisy import OptiProtesNoisyComp
from .opti_sspts import Optisspts


def opti_get():
    optis = []
    optis.append(OptiNB)
    optis.append(OptiOPO)
    optis.append(OptiOptimatt)
    optis.append(OptiPortfolio)
    optis.append(OptiProtes)
    optis.append(OptiDiPTS)
    optis.append(OptiPSO)
    optis.append(OptiSPSA)
    optis.append(OptiTTOpt)
    optis.append(OptiNoisyPTS)
    opti.append(OptiProtesNoisyComp)
    opti.append(Optisspts)
    return optis