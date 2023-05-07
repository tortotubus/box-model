"""
=================================
Model (:mod:`boxmodel.model`)
=================================

.. currentmodule:: boxmodel.model

A short description.
====================

.. autosummary::
   :toctree: generated/

   base   -- General purpose base classes which can be built upon
   viewer -- Viewers for different models
   models -- Different models derived from base models
"""

from .viewer import (BoxModelViewer, MultipleBoxModelViewer)
from .base import (BoxModel, MultipleBoxModel, 
                    BoxModelSolution, MultipleBoxModelSolution, 
                    DepositSolution, MultipleDepositSolution)
from .models import (BoxModelWithConcentration, MultipleBoxModelWithConcentration)