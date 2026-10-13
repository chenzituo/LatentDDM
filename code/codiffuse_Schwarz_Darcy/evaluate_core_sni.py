"""Canonical entry point for Local/Core-initialized owned-core Darcy SNI.

The implementation is retained in its historical module for checkpoint and
import compatibility.  New experiments should use this name so the executable
contract is unambiguous: sweep zero is Local/Core and every sweep writes
disjoint owned cores.
"""

from .evaluate_weighted_pou_sni_initializations import main


if __name__ == "__main__":
    main()
