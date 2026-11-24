"""Import aliases for stored model configurations."""

from .causalseqgan import CausalSeqGAN, CausalSeqGANConfig

SCMGenerator = CausalSeqGAN
SCMGeneratorConfig = CausalSeqGANConfig

__all__ = ["SCMGenerator", "SCMGeneratorConfig"]
