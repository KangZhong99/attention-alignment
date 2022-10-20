import torch.nn.functional as functional
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
from fairseq.criterions import register_criterion
import logging
import torch
import math
from fairseq import metrics, utils

@register_criterion('myloss')
class Myloss(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        lable_smoothing,
        flag_loss_weight,
        tau,
    ):
        super(),__init__(task,  sentence_avg, lable_smoothing)
        self.flag_loss_weight=flag_loss_weight
        self.tau=tau
    
    def add_args(parser):
        parser.add_argument('--mask-loss-weight',default=0.15, type=float, help='alpha')
        parser.add_argument('--tau',default=10000.,type=float, help='tau')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
    
    def forward(self, model, sample, reduce=True,show=False, step=0):
        net_output, net_output_flag = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        src_len=net_output[-1]["flag"][0].size()[-1]
        crs_attn=net_output[-1]['crs_attn'][0]
        flag=net_output[-1]['flag'][0]
        num_layers, num_heads, bsz, tgt_len, src_len=flag.size()
        assert flag.size()==crs_attn.size()

        aligned_attn = torch.mul(crs_attn, torch.exp(flag)).clone().detach()
        crs_loss=F.cosine_similarity(crs_attn, aligned_attn,dim=-1) #layers, heads,bsz, tgt_en
        crs_loss=crs_loss.sum()/(num_layers*num_heads)

        flag_loss, _ self.compute_loss(model, net_output_flag, sample, reduce=reduce)
        p_norm = torch.norm(1-net_output[-1]["flag"][0], p=2)/src_len
        flag_loss_final = -mask_loss+self.flag_loss_weight*p_norm
        logging_output = {
            "loss": loss.data,
            "flag_loss": flag_loss.data,
            "p2":p_norm.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "num_attn_loss": bsz*tgt_len
        }
    
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        # print(logging_outputs)
        flag_loss_sum = sum(log.get('flag_loss', 0) for log in logging_outputs)
        # mask_loss_final_sum = sum(log.get('mask_loss_final', 0) for log in logging_outputs)
        p_sum = sum(log.get('p2', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)


        metrics.log_scalar('flag_loss', flag_loss_sum / sample_size / math.log(2), sample_size, round=6)
        # metrics.log_scalar('mask_loss_final', mask_loss_final_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('p_2', p_sum / sample_size, sample_size, round=5)


