"""Compatibility entry point for the unified production-evidence audit.

The former implementation only checked whether the three Patch Transformer
checkpoints were independent global ViTs.  That narrow check could still pass
while figures, tables, method identity, timing, or the selected checkpoints
had drifted.  Keep this filename for existing workflows, but route every call
through :mod:`audit_submission_evidence`, which verifies the complete
no-POU/owned-core submission contract and the production manifest hashes.

The default remains release-strict.  During the validation-gated Airfoil run,
``--allow-pending-airfoil`` is the explicit development-only escape hatch.
"""

from audit_submission_evidence import main


if __name__ == "__main__":
    main()
