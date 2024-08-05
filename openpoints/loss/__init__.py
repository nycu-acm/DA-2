from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .distill_loss import  DistillLoss
from .build import build_criterion_from_cfg
from .sw_variants import ASW, SWD, GenSW, MaxSW
