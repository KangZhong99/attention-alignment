from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig
from fairseq.optim import register_optimizer
from omegaconf import II, DictConfig

@register_optimizer("myadam", dataclass=FairseqAdamConfig)
class MyFairseqAdam(FairseqAdam):
	def __init__(self, cfg: DictConfig, params):
		super().__init__(cfg, params)
	def backward(self, loss, retain_graph=False):
		loss.backward(retain_graph=retain_graph)

