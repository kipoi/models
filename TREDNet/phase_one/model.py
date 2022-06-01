# -*- coding: utf-8 -*-
"""
@author: okurman
"""

import numpy as np
from kipoi.model import KerasModel

class PhaseOneModel(KerasModel):
    def __init__(self, weights, arch):
        super().__init__(weights=weights, arch=arch)
