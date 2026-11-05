""""
Both 3D_bssfp_sequence.ipynb (builds/writes the .seq) and
Simulate.ipynb (imports the .seq and simulates) import these values, so the
phantom, the sequence and the downstream recon/analysis can never disagree on
grid size, readout geometry or timing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BSSFPParams:
    # ---- grid -----------------------------------------------------------
    FOV: float = 250e-3            # isotropic FOV [m]
    n: int = 64                    # phantom + recon matrix (n^3)
    N_x: int = 64                  # readout samples / spoke  (== n)
    N_channels: int = 8            # coil elements
    # ---- readout geometry -------------------------------------------------
    N_readout: int = 384           # echoes per inversion segment
    N_segments: int = 25           # inversion segments (25*384 = 9600 spokes)
    N_dummy: int = 0
    # ---- timing -----------------------------------------------------------
    slice_thickness: float = 64e-3
    TI_START: float = 20e-3        # TI of the first echo
    TI_inter: float = 4.88e-3      # recovery delay between echoes
    TR_echo: float = 7.8e-3
    P_echo: float = 12.6e-3
    TD: float = 2000e-3            # recovery delay at end of each segment
    TE: float = 3.98e-3 
    # ---- RF ---------------------------------------------------------------
    TE: float = 3.98e-3            # echo time
    alpha_deg: float = 60.0        # bSSFP flip angle
    rf_spoiling_inc: float = 180.0  # steady-state RF phase cycle per echo
    # ---- simulation -------------------------------------------------------
    spin_count: int = 64           # isochromats per voxel (simulation only)

    # ---- derived ----------------------------------------------------------
    @property
    def n_adc(self) -> int:
        """Total acquired ADC samples (all coils interleaved)."""
        return self.N_segments * self.N_readout * self.N_x

    def effective_tis(self, TR_echo: float) -> np.ndarray:
        """Effective TI of each echo index p = 0..N_readout-1.

        The train starts at TI_START and each echo index advances by one echo
        spacing P = TR_echo + TI_inter (TR_echo = duration of one GRE block).
        """
        P = TR_echo + self.TI_inter
        p = np.arange(self.N_readout)
        return self.TI_START + p * P


# Tissue ground truth (as defined in load_brain_phantom):
#   label 1 -> CSF: PD 1.0, T1 4.0 s, T2 0.5 s
#   label 2 -> GM : PD 0.8, T1 1.3 s, T2 0.08 s
#   label 3 -> WM : PD 0.7, T1 0.9 s, T2 0.07 s
#   T2* = 0.04 s everywhere
TISSUES = {
    "WM":  dict(T1=0.9, T2=0.07, PD=0.7),
    "GM":  dict(T1=1.3, T2=0.08, PD=0.8),
    "CSF": dict(T1=4.0, T2=0.5,  PD=1.0),
}
T2STAR = 0.04
NAMES = ("WM", "GM", "CSF")

# one shared instance
PARAMS = BSSFPParams()
