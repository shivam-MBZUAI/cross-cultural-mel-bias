"""
FairAudioBench Models Package
Contains implementations of all six audio front-ends for bias evaluation
"""

from .frontends import (
    BaseFrontEnd,
    StandardMelSpectrogramFrontEnd,
    ERBScaleFrontEnd, 
    GammatoneFilterBankFrontEnd,
    CochlearFilterBankFrontEnd,
    BarkScaleFrontEnd,
    LearnableMelFrontEnd,
    create_frontend,
    get_model_summary
)

__all__ = [
    'BaseFrontEnd',
    'StandardMelSpectrogramFrontEnd',
    'ERBScaleFrontEnd',
    'GammatoneFilterBankFrontEnd', 
    'CochlearFilterBankFrontEnd',
    'BarkScaleFrontEnd',
    'LearnableMelFrontEnd',
    'create_frontend',
    'get_model_summary'
]

__version__ = "1.0.0"
