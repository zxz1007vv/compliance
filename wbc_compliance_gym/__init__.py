

import os

WBC_COMPLIANCE_ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
WBC_COMPLIANCE_ENVS_DIR = os.path.join(
    WBC_COMPLIANCE_ROOT_DIR, "wbc_compliance_gym", "envs"
)

__all__ = ["WBC_COMPLIANCE_ENVS_DIR", "WBC_COMPLIANCE_ROOT_DIR"]
