import torch
import torch.nn.functional as F
from torch import nn
import torchvision

import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report

from torch.nn import Sequential, Linear, ReLU

from .utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .position_encoding import build_position_encoding
from .transformer import build_transformer, TransformerDecoder, TransformerDecoderLayer

from metrics.composed_loss import ComposedLoss, ComposedPatternLoss

class GarmentDETRv6(nn.Module):
    def __init__(self, backbone, panel_transformer, num_panel_queries, num_edges, num_joints, **edge_kwargs):
        super().__init__()
        self.backbone = backbone
        
        self.num_panel_queries = num_panel_queries
        self.num_joint_queries = num_joints
        self.panel_transformer = panel_transformer

        self.hidden_dim = self.panel_transformer.d_model

        self.panel_embed = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 2)

        self.panel_joints_query_embed = nn.Embedding(self.num_panel_queries + self.num_joint_queries, self.hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.panel_rt_decoder = MLP(self.hidden_dim, self.hidden_dim, 7, 2)
        self.joints_decoder = MLP(self.hidden_dim, self.hidden_dim, 6, 2)

        self.num_edges = num_edges
        self.num_edge_queries = self.num_panel_queries * num_edges
        self.edge_kwargs = edge_kwargs["edge_kwargs"]


        self.panel_decoder = MLP(self.hidden_dim, self.hidden_dim, self.num_edges * 4, 2)
        self.edge_query_mlp = MLP(self.hidden_dim + 4, self.hidden_dim, self.hidden_dim, 1)

        self.build_edge_decoder(self.hidden_dim, self.edge_kwargs["nheads"], 
                                self.hidden_dim, self.edge_kwargs["dropout"], 
                                "relu", self.edge_kwargs["pre_norm"], 
                                self.edge_kwargs["dec_layers"])
        
        self.edge_embed = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 2)
        self.edge_cls = MLP(self.hidden_dim, self.hidden_dim // 2, 1, 2)
        self.edge_decoder = MLP(self.hidden_dim, self.hidden_dim, 4, 2)

    def build_edge_decoder(self, d_model, nhead, dim_feedforward, dropout, activation, normalize_before, num_layers):
        edge_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.edge_trans_decoder = TransformerDecoder(edge_decoder_layer, num_layers, decoder_norm, return_intermediate=True)
        self._reset_parameters()
    
    def _reset_parameters(self, ):
        for p in self.edge_trans_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, samples, gt_stitches=None, gt_edge_mask=None, return_stitches=False):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, panel_pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        B = src.shape[0]
        assert mask is not None
        panel_joint_hs, panel_memory, _ = self.panel_transformer(self.input_proj(src), mask, self.panel_joints_query_embed.weight, panel_pos[-1])
        panel_joint_hs = self.panel_embed(panel_joint_hs)
        panel_hs = panel_joint_hs[:, :, :self.num_panel_queries, :]
        joint_hs = panel_joint_hs[:, :, self.num_panel_queries:, :]
        output_panel_rt = self.panel_rt_decoder(panel_hs)

        output_rotations = output_panel_rt[:, :, :, :4]
        output_translations = output_panel_rt[:, :, :, 4:]

        out = {"rotations": output_rotations[-1], 
               "translations": output_translations[-1]}
        
        output_joints = self.joints_decoder(joint_hs)
        out.update({"smpl_joints": output_joints[-1]})

        edge_output = self.panel_decoder(panel_hs)[-1].view(B, self.num_panel_queries, self.num_edges, 4)
        edge_query = self.edge_query_mlp(torch.cat((panel_joint_hs[-1, :, :self.num_panel_queries, :].unsqueeze(2).expand(-1, -1, self.num_edges, -1), edge_output), dim=-1)).reshape(B, -1, self.hidden_dim).permute(1, 0, 2)

        tgt = torch.zeros_like(edge_query)
        memory = panel_memory.view(B, self.hidden_dim, -1).permute(2, 0, 1)
        edge_hs = self.edge_trans_decoder(tgt, memory, 
                                          memory_key_padding_mask=mask.flatten(1), 
                                          query_pos=edge_query).transpose(1, 2)
        
        output_edge_embed = self.edge_embed(edge_hs)[-1]

        output_edge_cls = self.edge_cls(output_edge_embed)
        output_edges = self.edge_decoder(output_edge_embed) + edge_output.view(B, -1, 4)

        out.update({"outlines": output_edges, "edge_cls": output_edge_cls})

        if gt_stitches is not None and gt_edge_mask is not None:
            # Reshape gt_stitches and gt_edge_mask to be 2D
            gt_stitches = gt_stitches.view(B, -1)
            gt_edge_mask = gt_edge_mask.view(B, -1)
            
            # Ensure gather indices are within the bounds of output_edge_embed
            max_index = output_edge_embed.shape[1]
            gt_stitches = gt_stitches.clamp(max=max_index - 1)

            gather_indices = gt_stitches.unsqueeze(-1).expand(-1, -1, output_edge_embed.shape[-1])
            edge_node_features = torch.gather(output_edge_embed, 1, gather_indices.long())
            
            mask_for_fill = gt_edge_mask.unsqueeze(-1).expand_as(edge_node_features) == 0
            edge_node_features = edge_node_features.masked_fill(mask_for_fill, 0)
        else:
            edge_node_features = output_edge_embed
        
        edge_norm_features = F.normalize(edge_node_features, dim=-1)
        edge_similarity = torch.bmm(edge_norm_features, edge_norm_features.transpose(1, 2))
        
        # Mask out diagonal
        mask = torch.eye(edge_similarity.shape[1], device=edge_similarity.device).repeat(edge_similarity.shape[0], 1, 1).bool()
        edge_similarity[mask] = -float("inf")
        out.update({"edge_similarity": edge_similarity})
        
        return out

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class StitchLoss(nn.Module):
    """
    Robust binary cross-entropy loss for stitch similarity matrix.
    Handles shape mismatches and prevents numerical instability by using a boolean mask.
    """
    def __call__(self, similarity_matrix, gt_matrix, gt_free_mask=None):
        num_queries = similarity_matrix.shape[1]
        
        # Ensure gt_matrix has the same dimensions as similarity_matrix
        if gt_matrix.shape[1] != num_queries:
            if gt_matrix.shape[1] > num_queries:
                gt_matrix = gt_matrix[:, :num_queries, :num_queries]
            else:
                padding_size = num_queries - gt_matrix.shape[1]
                gt_matrix = F.pad(gt_matrix, (0, padding_size, 0, padding_size), "constant", 0)

        # Create a boolean mask to select valid elements for loss calculation
        valid_mask = torch.ones_like(similarity_matrix, dtype=torch.bool)

        if gt_free_mask is not None:
            if gt_free_mask.dim() > 2:
                gt_free_mask = gt_free_mask.reshape(gt_free_mask.shape[0], -1)

            batch_size, num_edges = gt_free_mask.shape
            
            if num_edges != num_queries:
                if num_edges < num_queries:
                    padding = torch.ones(batch_size, num_queries - num_edges, dtype=torch.bool, device=gt_free_mask.device)
                    gt_free_mask = torch.cat([gt_free_mask, padding], dim=1)
                else:
                    gt_free_mask = gt_free_mask[:, :num_queries]

            # An element is invalid if its row or column corresponds to a free edge
            invalid_row_mask = gt_free_mask.unsqueeze(2)  # (B, N, 1)
            invalid_col_mask = gt_free_mask.unsqueeze(1)  # (B, 1, N)
            invalid_mask = invalid_row_mask | invalid_col_mask
            valid_mask &= ~invalid_mask

        # Also mask out the diagonal
        diag_mask = ~torch.eye(num_queries, device=similarity_matrix.device, dtype=torch.bool).unsqueeze(0)
        valid_mask &= diag_mask

        gt_matrix = gt_matrix.float()

        # Select only the valid elements for loss calculation
        valid_preds = similarity_matrix[valid_mask]
        valid_targets = gt_matrix[valid_mask]
        
        # Handle case where there are no valid elements to prevent nan loss
        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=similarity_matrix.device, requires_grad=True), torch.tensor(1.0, device=similarity_matrix.device)

        loss = F.binary_cross_entropy_with_logits(valid_preds, valid_targets)
        
        # Accuracy calculation on valid elements
        accuracy = ((F.sigmoid(valid_preds) > 0.5) == (valid_targets > 0.5)).float().mean()
        
        return loss, accuracy

class SetCriterionWithOutMatcher(nn.Module):

    def __init__(self, data_config, in_config={}):
        super().__init__()
        self.config = {}
        self.config['loss'] = {
            'loss_components': ['shape', 'loop', 'rotation', 'translation'],
            'quality_components': ['shape', 'discrete', 'rotation', 'translation'],
            'panel_origin_invariant_loss': False,
            'loop_loss_weight': 1.,
            'stitch_tags_margin': 0.3,
            'epoch_with_stitches': 10000, 
            'stitch_supervised_weight': 0.1,
            'stitch_hardnet_version': False,
            'panel_origin_invariant_loss': True
        }

        self.config['loss'].update(in_config)

        self.composed_loss = ComposedPatternLoss(data_config, self.config['loss'])
        self.stitch_loss = StitchLoss()
    
    def forward(self, outputs, ground_truth, names=None, epoch=1000):

        b, q = outputs["outlines"].shape[0], outputs["rotations"].shape[1]
        outputs["outlines"] = outputs["outlines"].view(b, q, -1, 4).contiguous()
        full_loss, loss_dict, _ = self.composed_loss(outputs, ground_truth, names, epoch)
        
        if "edge_cls" in outputs and 'lepoch' in self.config['loss'] and epoch >= self.config['loss']['lepoch']:
            if epoch == -1:
                st_edge_precision, st_edge_recall, st_edge_f1_score, st_precision, st_recall, st_f1_score, st_adj_precs, st_adj_recls, st_adj_f1s = self.prediction_stitch_rp(outputs, ground_truth)
                loss_dict.update({"st_edge_prec": st_edge_precision,
                                  "st_edge_recl": st_edge_recall,
                                  "st_edge_f1s": st_edge_f1_score,
                                  "st_prec": st_precision,
                                  "st_recl": st_recall,
                                  "st_f1s": st_f1_score,
                                  "st_adj_precs": st_adj_precs, 
                                  "st_adj_recls": st_adj_recls, 
                                  "st_adj_f1s": st_adj_f1s})
            
            # --- Simplified and Robust Loss Calculation ---
            edge_cls_gt = (~ground_truth["free_edges_mask"].view(b, -1)).float().to(outputs["edge_cls"].device)
            edge_cls_loss = torchvision.ops.sigmoid_focal_loss(outputs["edge_cls"].squeeze(-1), edge_cls_gt, reduction="mean")
            
            full_loss = full_loss * 5
            loss_dict.update({"stitch_cls_loss": 0.5 * edge_cls_loss})
            edge_cls_acc = ((F.sigmoid(outputs["edge_cls"].squeeze(-1)) > 0.5) == edge_cls_gt).sum().float() / (edge_cls_gt.shape[0] * edge_cls_gt.shape[1])
            loss_dict.update({"stitch_edge_cls_acc": edge_cls_acc})
            full_loss += loss_dict["stitch_cls_loss"]
            
            # Unconditional call to the robust stitch loss
            stitch_loss, stitch_acc = self.stitch_loss(outputs["edge_similarity"], ground_truth["stitch_adj"], ground_truth["free_edges_mask"])
            
            if stitch_loss is not None and stitch_acc is not None:
                loss_dict.update({"stitch_loss": 0.05 * stitch_loss, "stitch_acc": stitch_acc})
                full_loss += loss_dict["stitch_loss"]

            if "smpl_joints" in ground_truth and "smpl_joints" in outputs:
                joints_loss = F.mse_loss(outputs["smpl_joints"], ground_truth["smpl_joints"])
                loss_dict.update({"smpl_joint_loss": joints_loss})
                full_loss += loss_dict["smpl_joint_loss"]
        
        return full_loss, loss_dict
    
    def prediction_stitch_rp(self, outputs, ground_truth):
        if "edge_cls" not in outputs:
            return [0], [0], [0], [0], [0], [0], [0], [0], [0]

        bs = outputs["outlines"].shape[0]
        st_edge_pres, st_edge_recls, st_edge_f1s, st_precs, st_recls, st_f1s, st_adj_precs, st_adj_recls, st_adj_f1s = [], [], [], [], [], [], [], [], []
        
        for b in range(bs):
            edge_cls_gt = (~ground_truth["free_edges_mask"][b]).flatten()
            edge_cls_pr = (F.sigmoid(outputs["edge_cls"][b].squeeze(-1)) > 0.5).flatten()
            
            # Add zero_division=0 to prevent crash on empty predictions
            cls_rept = classification_report(edge_cls_gt.detach().cpu().numpy(), edge_cls_pr.detach().cpu().numpy(), labels=[0,1], zero_division=0)
            strs = cls_rept.split("\n")[3].split()
            st_edge_precision, st_edge_recall, st_edge_f1_score = float(strs[1]), float(strs[2]), float(strs[3])

            st_cls_pr = (F.sigmoid(outputs["edge_similarity"][b].squeeze(-1)) > 0.5).flatten()
            stitch_cls_rept = classification_report(st_cls_pr.detach().cpu().numpy(), ground_truth["stitch_adj"][b].flatten().detach().cpu().numpy(), labels=[0, 1], zero_division=0)
            strs = stitch_cls_rept.split("\n")[3].split()
            st_adj_edge_precision, st_adj_edge_recall, st_adj_edge_f1_score = float(strs[1]), float(strs[2]), float(strs[3])

            st_adj_precs.append(st_adj_edge_precision)
            st_adj_recls.append(st_adj_edge_recall)
            st_adj_f1s.append(st_adj_edge_f1_score)

            edge_similarity = outputs["edge_similarity"][b]
            simi_matrix = torch.triu(edge_similarity, diagonal=1)
            stitches = []
            num_stitches = edge_cls_pr.nonzero().shape[0] // 2
            
            for i in range(num_stitches):
                if torch.max(simi_matrix) == -float("inf"):
                    break
                index = (simi_matrix == torch.max(simi_matrix)).nonzero()
                stitches.append((index[0, 0].cpu().item(), index[0, 1].cpu().item()))
                simi_matrix[index[0, 0], :] = -float("inf")
                simi_matrix[index[0, 1], :] = -float("inf")
                simi_matrix[:, index[0, 0]] = -float("inf")
                simi_matrix[:, index[0, 1]] = -float("inf")
            
            st_precision, st_recall, st_f1_score = SetCriterionWithOutMatcher.set_precision_recall(stitches, ground_truth["stitches"][b])

            st_edge_pres.append(st_edge_precision)
            st_edge_recls.append(st_edge_recall)
            st_edge_f1s.append(st_edge_f1_score)
            st_precs.append(st_precision)
            st_recls.append(st_recall)
            st_f1s.append(st_f1_score)

        return st_edge_pres, st_edge_recls, st_edge_f1s, st_precs, st_recls, st_f1s, st_adj_precs, st_adj_recls, st_adj_f1s
    
    @staticmethod
    def set_precision_recall(pred_stitches, gt_stitches):
        def elem_eq(a, b):
            return (a[0] == b[0] and a[1] == b[1]) or (a[0] == b[1] and a[1] == b[0])
        
        gt_stitches = gt_stitches.transpose(0, 1).cpu().detach().numpy()
        true_pos = 0
        
        # Filter out padding in ground truth
        valid_gt_stitches = [tuple(s) for s in gt_stitches if s[0] != -1 and s[1] != -1]
        
        for pstitch in pred_stitches:
            for gstitch in valid_gt_stitches:
                if elem_eq(pstitch, gstitch):
                    true_pos += 1
                    break # Move to next predicted stitch
        
        false_pos = len(pred_stitches) - true_pos
        false_neg = len(valid_gt_stitches) - true_pos

        precision = true_pos / (true_pos + false_pos + 1e-6)
        recall = true_pos / (true_pos + false_neg + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        return precision, recall, f1_score
    
    def with_quality_eval(self, ):
        if hasattr(self.composed_loss, "with_quality_eval"):
            self.composed_loss.with_quality_eval = True
    
    def print_debug(self):
        self.composed_loss.debug_prints = True
    
    def train(self, mode=True):
        super().train(mode)
        self.composed_loss.train(mode)
    
    def eval(self):
        super().eval()
        if isinstance(self.composed_loss, object):
            self.composed_loss.eval()
            
def build(args):
    num_classes = args["dataset"]["max_pattern_len"]
    try:
        if torch.cuda.is_available():
            devices = torch.device(f"cuda:{args['trainer']['devices'][0]}")
        else:
            devices = torch.device("cpu")
    except (KeyError, IndexError):
        devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Building model on device: {devices}")

    backbone = build_backbone(args)
    panel_transformer = build_transformer(args)

    model = GarmentDETRv6(backbone, panel_transformer, num_classes, 14, 22, edge_kwargs=args["NN"])

    criterion = SetCriterionWithOutMatcher(args["dataset"], args["NN"]["loss"])
    criterion.to(devices)
    return model, criterion