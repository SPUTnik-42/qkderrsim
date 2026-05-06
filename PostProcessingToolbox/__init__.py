from .cascade_refactored import CascadeClientProtocol
from .winnow_refactored import WinnowClientProtocol
from .polar_codes import PolarClientProtocol
from .privacy_amplification import PrivacyAmplification
from .oracle import ParityOracle
from .prng import PRNG

# Try to import LDPC protocols (optional - require TensorFlow/Sionna)
NR_LDPC_Standard_ClientProtocol = None
LDPC_RateAdaptive_ClientProtocol = None

try:
    from .nr_ldpc_standard import NR_LDPC_Standard_ClientProtocol
except ImportError as e:
    print(f"[WARNING] NR-LDPC Standard unavailable (missing TensorFlow/Sionna): {e}")

try:
    from .ldpc_rateadaptive import LDPC_RateAdaptive_ClientProtocol
except ImportError as e:
    print(f"[WARNING] LDPC Rate Adaptive unavailable (missing TensorFlow/Sionna): {e}")

__all__ = [
    "CascadeClientProtocol",
    "WinnowClientProtocol",
    "PolarClientProtocol",
    "NR_LDPC_Standard_ClientProtocol",
    "LDPC_RateAdaptive_ClientProtocol",
    "PrivacyAmplification",
    "ParityOracle",
    "PRNG"
]
