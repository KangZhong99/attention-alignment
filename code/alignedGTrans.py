
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models.transformer import (
    TransformerModel,
    TransformerDecoder,
    transformer_doc_base,
    tokens2tags
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from fairseq.modules import (
    AdaptiveSoftmax,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

@register_model('aligned_transformer')
class AlignedTransformerMdel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
    
    def add_args(parser):
        TransformerModel.add_args(parser)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return AlignedTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            loss_type='nmt'
        )
        decoder_out_flag = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            loss_type='flag'
        )
        return decoder_out,decoder_out_flag




class AlignedTransformerDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super.__init__(args=args, dictionary=dictionary, embed_tokens=embed_tokens, no_encoder_attn=no_cross_attention)

    def build_decoder_layer(self, args, no_encoder_attn, dec_add_global_attn, crs_add_global_attn):
        return AlignedTransformerDecoderLayer(args, no_encoder_attn, dec_add_global_attn, crs_add_global_attn)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        loss_type='nmt'
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            loss_type=loss_type
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        loss_type='nmt'
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            loss_type=loss_type
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        loss_type='nmt'
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed tags
        decoder_tags = None
        if self.self_partial_mode or self.cross_partial_mode:
            prev_output_tags = tokens2tags(self.dictionary, prev_output_tokens, self.eod)
            decoder_tags = prev_output_tags

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if decoder_tags is not None:
                decoder_tags = decoder_tags[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            padding_mask = prev_output_tokens.eq(self.padding_idx)

        # Guangsheng Bao: the partial mode enables the Group Attention (code name: local_attn)
        # Here we generate attention mask for Group Attention and Global Attention according to group tags
        encoder_local_mask = None
        local_attn_mask = None
        global_attn_mask = None
        # cross attention
        if self.cross_partial_mode:
            encoder_local_mask = encoder_out.encoder_tags.unsqueeze(1) != decoder_tags.unsqueeze(2)
            encoder_local_mask &= 0 != decoder_tags.unsqueeze(2)
        # self attention
        if self.self_partial_mode:
            local_attn_mask = prev_output_tags.unsqueeze(1) != decoder_tags.unsqueeze(2)
            local_attn_mask &= 0 != decoder_tags.unsqueeze(2)

        # add to the triangle self attention mask
        if incremental_state is None and not full_context_alignment:
            self_attn_mask = self.buffered_future_mask(x).bool()
            if local_attn_mask is not None:
                local_attn_mask |= self_attn_mask.unsqueeze(0)
            else:
                local_attn_mask = self_attn_mask
            global_attn_mask = self_attn_mask

        # decoder layers
        attn: List[Optional[Tensor]] = []
        inner_states: List[Optional[Tensor]] = [x]
        flag=[]
        crs_attn=[]
        for idx, layer in enumerate(self.layers):
            x, layer_attn, layer_flag_attn = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_local_mask,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                local_attn_mask=local_attn_mask,
                global_attn_mask=global_attn_mask,
                padding_mask=padding_mask,
                loss_type=loss_type
            )
            inner_states.append(x)
            if layer_attn is not None:
                attn.append(layer_attn)
            #kangzhong:
            if len(layer_flag_attn)>0:
                flag.append(layer_flag_attn[0])
                crs_attn.append(layer_flag_attn[1])


        attn = utils.average_layers_attn(attn, reduce=torch.sum)
        if encoder_out.encoder_attn is not None:
            attn.update(encoder_out.encoder_attn)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        #kangzhong:
        flag=torch.stack(flag,dim=0) # layer, num_heads, bsz, tgt_len, src_len
        crs_attn=torch.stack(crs_attn) #  the same dim with flaf
        return x, {"attn": attn, "inner_states": inner_states, 'flag':[flag], 'crs_attn':[crs_attn]}


class AlignedTransformerDecoderLayer(TransformerDecoderLayer):
     """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False,
        dec_add_global_attn=False, crs_add_global_attn=False,
        add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.dropout
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.need_attn_entropy = getattr(args, 'doc_attn_entropy', False)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        # Guangsheng Bao:
        # XXX_attn_local is for local sentence (Group Attention), while XXX_attn_global is for global (Global Attention)
        assert no_encoder_attn == False
        self.self_attn_local = self.build_self_attention(self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
        self.encoder_attn_local = self.build_encoder_attention(self.embed_dim, args.decoder_attention_heads, args)
        if dec_add_global_attn:
            self.self_attn_global = self.build_self_attention(self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
            self.self_attn_gate = nn.Sequential(Linear(self.embed_dim * 2, self.embed_dim), nn.Sigmoid())
        if crs_add_global_attn:
            #kangzhong:
            self.context_detector=self.build_context_detector(args)
            self.encoder_attn_global = self.build_encoder_attention_with_flag(self.embed_dim, args.decoder_attention_heads, args)
            self.encoder_attn_gate = nn.Sequential(Linear(self.embed_dim * 2, self.embed_dim), nn.Sigmoid())

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)

        self.fc1 = self.build_fc1(
            self.embed_dim, args.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.normalize_before = args.decoder_normalize_before

        self.onnx_trace = False
    
    def build_encoder_attention_with_flag(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return AlignedMultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
    def build_context_detector(self, args):
        return CD(embed_dim=args.decoder_embed_dim, 
                    num_heads=args.decoder_attention_heads,
                    dropout=args.attention_dropout)
    
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_local_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        local_attn_mask: Optional[torch.Tensor] = None,
        global_attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        loss_type='nmt'
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        attn = {}
        #kangzhong:
        flag_attn=[]

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x_local, attn_local = self.self_attn_local(
            query=x, key=x, value=x,
            key_padding_mask=padding_mask,
            incremental_state=incremental_state,
            attn_mask=local_attn_mask,
            need_weights=self.need_attn_entropy,
        )
        if attn_local is not None:
            attn['decoder_self_local'] = utils.attn_entropy(attn_local, mean_dim=1)

        # for partial mode, we combine local and global attention
        if getattr(self, 'self_attn_global', None) is not None:
            x_global, attn_global = self.self_attn_global(
                query=x, key=x, value=x,
                key_padding_mask=padding_mask,
                incremental_state=incremental_state,
                attn_mask=global_attn_mask,
                need_weights=self.need_attn_entropy,
            )
            if attn_global is not None:
                attn['decoder_self_global'] = utils.attn_entropy(attn_global, mean_dim=1)
            # merge with local
            g = self.self_attn_gate(torch.cat([x_local, x_global], dim=-1))
            x = x_local * g + x_global * (1 - g)
        else:
            x = x_local

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn_local is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x_local, attn_local = self.encoder_attn_local(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                attn_mask=encoder_local_mask,
                need_weights=self.need_attn_entropy,
            )
            if attn_local is not None:
                attn['decoder_cross_local'] = utils.attn_entropy(attn_local, mean_dim=1)

            # for partial mode, we combine local and global attention
            if getattr(self, 'encoder_attn_global', None) is not None:
                #kangzhong
                flag=self.context_detector(x,encoder_out,encoder_out, encoder_padding_mask)
                x_global, attn_global crs_attn= self.encoder_attn_global(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=self.need_attn_entropy,
                    flag=flag,
                    loss_type=loss_type,
                )
                if attn_global is not None:
                    attn['decoder_cross_global'] = utils.attn_entropy(attn_global, mean_dim=1)
                # merge with local
                g = self.encoder_attn_gate(torch.cat([x_local, x_global], dim=-1))
                x = x_local * g + x_global * (1 - g)

                #kangzhong:
                tgt_len,bsz,_=x.size()
                src_len=encoder_out.size(0)
                flag=flag.view(bsz,-1,tgt_len,src_len).transpose(0,1) #num_heads bsz tgt_len src_len
                flag_attn.append(flag)
                flag_attn.append(crs_attn)

            else:
                x = x_local

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            assert False
            saved_state_local = self.self_attn_local._get_input_buffer(incremental_state)
            assert saved_state_local is not None
            if padding_mask is not None:
                self_attn_state = [
                    saved_state_local["prev_key"],
                    saved_state_local["prev_value"],
                    saved_state_local["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state_local["prev_key"], saved_state_local["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, flag_attn



class AlignedMultiheadAttention(MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        qdim=None,
        kdim=None,
        vdim=None,
        odim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super.__init__(embed_dim,
        num_heads,
        qdim=qdim,
        kdim=kdim,
        vdim=vdim,
        odim=odim,
        dropout=dropout,
        bias=bias,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
        self_attention=self_attention,
        encoder_decoder_attention=encoder_decoder_attention,
        q_noise=q_noise,
        qn_block_size=qn_block_size,)
    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        flag=None,
        loss_type='nmt'
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        # Guangsheng Bao: allow query dim to be different from embed_dim
        # assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, self.qdim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            # apply attn_mask to attn_weights
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1), float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf")
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        #kangzhong:
        if loss_type=='nmt':
            pass
        elif loss_type=='flag':
            attn_weights_float=torch.add(
                attn_weights_float,
                +torch.mul(torch.mean(attn_weights_float,-1,True),flag)
            )


        attn_weights = attn_weights_float.type_as(attn_weights)

        #kangzhong:
        crs_attn=attn_weights.view(bsz, self.num_heads, tgt_len, src_len).transpose(0,1)


        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, crs_attn

            
        


class CD(MultiheadAttention):
     def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        kdim=None,
        vdim=None,
    ):
        super().__init__(embed_dim, num_heads, kdim=kdim, vdim=vdim, dropout=dropout)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """

        tgt_len, bsz, embed_dim = query.size()
        # Guangsheng Bao: allow query dim to be different from embed_dim
        # assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, self.qdim]
    
        q = self.q_proj(query)
        k = self.k_proj(key)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf")
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        #attn_weights_float = utils.softmax( attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights_float = torch.tanh(attn_weights)
        attn_weights = attn_weights_float.type_as(attn_weights)

        return attn_weights


@register_model_architecture('aligned_transformer','aligned_transformer')
def my_hyperparameters(args):
    transformer_doc_base(args)