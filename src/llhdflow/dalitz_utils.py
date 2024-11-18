"""
Adapted from
https://gitlab.cern.ch/poluekt/TensorFlowAnalysis/-/blob/master/TensorFlowAnalysis/PhaseSpace/DalitzPhaseSpace.py
"""
import numpy as np

__all__ = ["dalitz_to_square", "square_to_dalitz"]


def __dir__():
    return __all__


def dalitz_to_square(
    sample: np.ndarray, md: float, ma: float, mb: float, mc: float
) -> np.ndarray:
    """
    Convert Dalitz to square between [0,1]

    D -> A B C

    Args:
        sample (``np.ndarray``): dalitz sample
        md (``float``): mother mass [GeV/c^2]
        ma (``float``): first daughter [GeV/c^2]
        mb (``float``): second daughter [GeV/c^2]
        mc (``float``): third daughter [GeV/c^2]

    Returns:
        ``np.ndarray``:
        transformed samples
    """
    x = MPrimeAC(sample, md, ma, mb, mc)
    y = ThetaPrimeAC(sample, md, ma, mb, mc)
    return np.stack([x, y], axis=1)


def square_to_dalitz(
    sample: np.ndarray, md: float, ma: float, mb: float, mc: float
) -> np.ndarray:
    """
    Convert Square data to Dalitz

    Args:
        sample (``np.ndarray``): square dalitz sample
        md (``float``): mother mass [GeV/c^2]
        ma (``float``): first daughter [GeV/c^2]
        mb (``float``): second daughter [GeV/c^2]
        mc (``float``): third daughter [GeV/c^2]

    Returns:
        ``np.ndarray``:
        Dalitz data
    """
    return FromSquareDalitzPlot(sample[:, 0], sample[:, 1], md, ma, mb, mc)


def Mac(sample: np.ndarray, md: float, ma: float, mb: float, mc: float) -> np.ndarray:
    """
    Get :math:`m^2_{ac}`

    Args:
        sample (``np.ndarray``): dalitz sample
        md (``float``): mother mass [GeV/c^2]
        ma (``float``): first daughter [GeV/c^2]
        mb (``float``): second daughter [GeV/c^2]
        mc (``float``): third daughter [GeV/c^2]

    Returns:
        ``np.ndarray``:
        :math:`m^2_{ac}`
    """
    return np.square([md, ma, mb, mc]).sum() - sample[:, 0] - sample[:, 1]


def ThetaPrimeAC(
    sample: np.ndarray, md: float, ma: float, mb: float, mc: float
) -> np.ndarray:
    """
    Square Dalitz plot variable theta'

    Args:
        sample (``np.ndarray``): dalitz sample
        md (``float``): mother mass [GeV/c^2]
        ma (``float``): first daughter [GeV/c^2]
        mb (``float``): second daughter [GeV/c^2]
        mc (``float``): third daughter [GeV/c^2]

    Returns:
        ``np.ndarray``:
        _description_
    """
    m2ac = Mac(sample, md, ma, mb, mc)
    return np.arccos(-CosHelicityAngleDalitz(m2ac, sample[:, 1], md, ma, mb, mc)) / np.pi


def CosHelicityAngleDalitz(
    m2ab: np.ndarray, m2bc: np.ndarray, md: float, ma: float, mb: float, mc: float
) -> np.ndarray:
    """
    Calculate cos(helicity angle) for set of two Dalitz plot variables
    m2ab, m2bc : Dalitz plot variables (inv. masses squared of AB and BC combinations)
    md : mass of the decaying particle
    ma, mb, mc : masses of final state particles
    """
    md2 = md**2
    ma2 = ma**2
    mb2 = mb**2
    mc2 = mc**2
    mab = np.sqrt(m2ab)
    eb = (m2ab - ma2 + mb2) / 2.0 / mab
    ec = (md2 - m2ab - mc2) / 2.0 / mab
    pb = np.sqrt(eb**2 - mb2)
    pc = np.sqrt(ec**2 - mc2)
    e2sum = (eb + ec) ** 2
    m2bc_max = e2sum - (pb - pc) ** 2
    m2bc_min = e2sum - (pb + pc) ** 2
    return (m2bc_max + m2bc_min - 2.0 * m2bc) / (m2bc_max - m2bc_min)


def MPrimeAC(sample, md, ma, mb, mc):
    """
    Square Dalitz plot variable m'
    m2ab : inv.-mass squared of A and B
    minab, maxab : the minimum and maximum allowed values for the invariant mass of A and B (typically, ma+mb and md-mc)
    """
    mac = np.sqrt(Mac(sample, md, ma, mb, mc))
    minac = ma + mc
    maxac = md - mb
    return np.arccos(2 * (mac - minac) / (maxac - minac) - 1.0) / np.pi


def FromSquareDalitzPlot(
    mprimeac: np.ndarray,
    thprimeac: np.ndarray,
    md: float,
    ma: float,
    mb: float,
    mc: float,
) -> np.ndarray:
    """
    sample: Given mprimeac and thprimeac, returns 2D tensor for (m2ab, m2bc).
    Make sure you don't pass in sqDP corner points as they lie outside phsp.
    """
    min_max = [(md - mb) ** 2, (ma + mc) ** 2]
    # maxac = (md - mb) ** 2
    # minac = (ma + mc) ** 2
    maxac = max(min_max)
    minac = min(min_max)

    msqsum = np.square([md, ma, mb, mc]).sum()

    m2AC = (
        0.25
        * (
            maxac**0.5 * np.cos(np.pi * mprimeac)
            + maxac**0.5
            - minac**0.5 * np.cos(np.pi * mprimeac)
            + minac**0.5
        )
        ** 2
    )

    m2AB = (
        0.5
        * (
            -(m2AC**2)
            + m2AC * ma**2
            + m2AC * mb**2
            + m2AC * mc**2
            + m2AC * md**2
            - m2AC
            * np.sqrt(
                (
                    m2AC * (m2AC - 2.0 * ma**2 - 2.0 * mc**2)
                    + ma**4
                    - 2.0 * ma**2 * mc**2
                    + mc**4
                )
                / m2AC
            )
            * np.sqrt(
                (
                    m2AC * (m2AC - 2.0 * mb**2 - 2.0 * md**2)
                    + mb**4
                    - 2.0 * mb**2 * md**2
                    + md**4
                )
                / m2AC
            )
            * np.cos(np.pi * thprimeac)
            - ma**2 * mb**2
            + ma**2 * md**2
            + mb**2 * mc**2
            - mc**2 * md**2
        )
        / m2AC
    )
    m2BC = msqsum - m2AC - m2AB
    return np.stack([m2AB, m2BC], axis=1)
