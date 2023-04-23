import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import numpy as np
import time
from util.misc import _get_clones
from .transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerCrossLayer

class ATFmodule(nn.Module):
    def __init__(self):
        super(ATFmodule, self).__init__()
        self.attetion_block = nn.Sequential(nn.Linear(256*2,256),
                                            nn.ReLU(),
                                            nn.Linear(256,256),
                                            nn.Sigmoid())
        self.mlp_layer = nn.Sequential(nn.Linear(256*2,256),
                                       nn.ReLU(),
                                       nn.Linear(256,256),
                                       nn.LayerNorm(256))
    def forward(self,x,y):
        att = self.attetion_block(torch.cat([x,y],dim=-1))
        ret = x + att * self.mlp_layer(torch.cat([x,y],dim=-1))
        return ret

class ATF(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.args = args

        if self.args.sharing_fusion_module:
            self.atfm = ATFmodule()
        else:
            self.atfm = _get_clones(ATFmodule(),self.args.dec_layers)

    def forward(self,task_feat,context,n):
        if self.args.sharing_fusion_module:
            return self.atfm(task_feat,context)
        return self.atfm[n](task_feat,context)
def make_mlp(dim_in,dim_out):
    module = nn.Sequential(nn.Linear(dim_in,dim_out),
                           nn.ReLU(),
                           nn.Linear(dim_out,dim_out),
                           nn.LayerNorm(dim_out)
                           )
    return module

class MURE(nn.Module):
    def __init__(self,dim=256,nhead=8,feeddim=2048):
        super(MURE, self).__init__()

        self.ternary = make_mlp(dim*3,dim)

        self.human_obj = make_mlp(dim*2,dim)
        self.human_rel = make_mlp(dim*2,dim)
        self.obj_rel = make_mlp(dim*2,dim)

        self.unary_self = TransformerEncoderLayer(dim,nhead,dim_feedforward=feeddim)
        self.pairwise_self = TransformerEncoderLayer(dim,nhead,dim_feedforward=feeddim)
        self.unary_cross = TransformerCrossLayer(dim,nhead,dim_feedforward=feeddim)
        self.pairwise_cross = TransformerCrossLayer(dim,nhead,dim_feedforward=feeddim)

        self.mc_gen = TransformerCrossLayer(dim,nhead,dim_feedforward=feeddim)

    def forward(self,output_sub,output_obj,output_rel,decoding_stuff=None):

        outputs_ternary = self.ternary((torch.cat([output_sub,output_obj,output_rel],dim=-1)))
        memory,tgt_mask,memory_mask,tgt_key_padding_mask,memory_key_padding_mask,pos = decoding_stuff

        # unary information encoding
        outputs_unary = self.unary_self(torch.stack([output_sub,output_obj,output_rel],dim=0).flatten(1,2)) \
                            .view(3,output_sub.size(0),output_sub.size(1),output_sub.size(2))
        outputs_tu = self.unary_cross(tgt=outputs_ternary.flatten(0,1).unsqueeze(0),memory=outputs_unary.flatten(1,2)).view(output_sub.size(0),output_sub.size(1),output_sub.size(2))
        # second information encoding

        human_obj_feat = self.human_obj(torch.cat([outputs_unary[0],outputs_unary[1]],dim=-1))
        human_rel_feat = self.human_rel(torch.cat([outputs_unary[0],outputs_unary[2]],dim=-1))
        obj_rel_feat = self.obj_rel(torch.cat([outputs_unary[1],outputs_unary[2]],dim=-1))

        outputs_pairwise = self.pairwise_self(torch.stack([human_obj_feat,human_rel_feat,obj_rel_feat],dim=0).flatten(1,2)) \
                                   .view(3,output_sub.size(0),output_sub.size(1),output_sub.size(2))


        outputs_tup = self.pairwise_cross(tgt=outputs_tu.flatten(0,1).unsqueeze(0),memory=outputs_pairwise.flatten(1,2)).view(output_sub.size(0),output_sub.size(1),output_sub.size(2))

        multiplex_context = self.mc_gen(outputs_tup,memory,tgt_mask=tgt_mask,
                                          memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask,
                                          pos=pos)

        return multiplex_context

class MUREN(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False,args=None):
        super().__init__()

        self.args=args
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before,return_attn=False)
        decoder_norm = nn.LayerNorm(d_model)


        self.decoder = TransformerDecoderThreeBranch(decoder_layer,num_dec_layers, decoder_norm, return_intermediate=return_intermediate_dec,args=args)


        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_human, query_obj, query_rel, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        query_human = query_human.unsqueeze(1).repeat(1, bs, 1)
        query_obj = query_obj.unsqueeze(1).repeat(1, bs, 1)
        query_rel = query_rel.unsqueeze(1).repeat(1, bs, 1)

        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        tgt_human = torch.zeros_like(query_human)
        tgt_obj = torch.zeros_like(query_obj)
        tgt_rel = torch.zeros_like(query_rel)

        out_human, out_obj, out_rel = self.decoder(tgt_human, tgt_obj, tgt_rel, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos_human = query_human, query_pos_obj= query_obj, query_pos_rel=query_rel, )

        out_human = out_human.transpose(1, 2)
        out_obj = out_obj.transpose(1, 2)
        out_rel = out_rel.transpose(1, 2)

        return out_human, out_obj, out_rel, memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerDecoderThreeBranch(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,args=None):
        super().__init__()
        self.args = args
        self.return_intermediate = return_intermediate
        self.args.sharing_fusion_module = False

        self.layers_human = _get_clones(decoder_layer, num_layers)
        self.layers_obj = _get_clones(decoder_layer, num_layers)
        self.layers_rel = _get_clones(decoder_layer, num_layers)
        self.norm_human, self.norm_obj, self.norm_rel = [copy.deepcopy(norm) for _ in range(3)]
        self.MURE = MURE()

        if self.args.dataset_file == 'vcoco':
            self.args.sharing_fusion_module = True

        self.aft_human = ATF(args)
        self.aft_obj = ATF(args)
        self.aft_rel = ATF(args)


    def forward(self, tgt_human, tgt_obj, tgt_rel, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos_human : Optional[Tensor] = None,
                query_pos_obj: Optional[Tensor] = None,
                query_pos_rel: Optional[Tensor] = None,
                ):
        output_human, output_obj, output_rel = tgt_human, tgt_obj, tgt_rel

        intermediate_human = []
        intermediate_obj = []
        intermediate_rel = []

        for layer_n, (layer_human, layer_obj, layer_rel) in enumerate(zip(self.layers_human, self.layers_obj, self.layers_rel)):
            output_human = layer_human(output_human, memory,tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=query_pos_human)

            output_obj = layer_obj(output_obj, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=query_pos_obj)

            output_rel = layer_rel(output_rel, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=query_pos_rel)

            multiplex_context = self.MURE(output_human,output_obj,output_rel,(memory,tgt_mask,memory_mask,tgt_key_padding_mask,memory_key_padding_mask,pos))

            output_human = self.aft_human(output_human,multiplex_context,layer_n)
            output_obj = self.aft_obj(output_obj,multiplex_context,layer_n)
            output_rel = self.aft_rel(output_rel,multiplex_context,layer_n)

            if self.return_intermediate:
                intermediate_human.append(self.norm_human(output_human))
                intermediate_obj.append(self.norm_obj(output_obj))
                intermediate_rel.append(self.norm_rel(output_rel))

        if self.return_intermediate:
            return torch.stack(intermediate_human), torch.stack(intermediate_obj), torch.stack(intermediate_rel)

        return output_human, output_obj, output_rel

def build_muren(args):
    return MUREN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        args=args
    )