import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from sg2im.layers import build_mlp
from sg2im.crn import RefinementNetwork
from sg2im.layout import boxes_to_layout, masks_to_layout
from sg2im.graph import GraphTripleConv, GraphTripleConvNet


class ReflectionModel(nn.Module):
    def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
                 gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5,
                 refinement_dims=(1024, 512, 256, 128, 64),
                 normalization='batch', activation='leakyrelu-0.2',
                 mask_size=None, mlp_normalization='none', layout_noise_dim=0,
                 **kwargs):
        super(ReflectionModel, self).__init__()

        if len(kwargs) > 0:
            print("WARNING: Model got unexpected kwargs ", kwargs)

        self.vocab = vocab
        self.image_size = image_size
        self.layout_noise_dim = layout_noise_dim

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])
        self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
        self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)

        if gconv_num_layers == 0:
            self.gconv = nn.Linear(embedding_dim, gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                'input_dim': embedding_dim,
                'output_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                'input_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)
        
        # Network for regressing bounding box using multilayer perceptual
        box_net_dim = 4
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)
        
        # Network for regress segmentation.
        self.mask_net = None
        if mask_size is not None and mask_size > 0:
            self.mask_net = self._build_mask_net(num_objs, gconv_dim, mask_size)
        
         #    rel_aux_layers >>??
        rel_aux_layers = [2 * embedding_dim + 8, gconv_hidden_dim, num_preds]
        self.rel_aux_net = build_mlp(
            rel_aux_layers, batch_norm=mlp_normalization)

        # Define cascaded refinement network.
        refinement_kwargs = {
            'dims': (gconv_dim + layout_noise_dim,) + refinement_dims,
          'normalization': normalization,
          'activation': activation,
        }
        self.refinement_net = RefinementNetwork(**refinement_kwargs)

    # private mask net.
    # TODO: seems to simple
    def _build_mask_net(self, num_objs, dim, mask_size):
        output_dim = 1
        layers, cur_size = [], 1
        while cur_size < mask_size:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.BatchNorm2d(dim))
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            cur_size *= 2
        if cur_size != mask_size:
            raise ValueError('Mask size must be a power of 2')
        layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, objs, triples, obj_to_img=None,
                boxes_gt=None, masks_gt=None):
        """ 
        Required Inputs:
        - objs: LongTensor of shape (O,) giving categories for all objects
        - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
          means that there is a triple (objs[s], p, objs[o])
        
        main Process: graph >>> graph conv >>> layout >>> CRN.

        Optional Inputs:
        - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
          means that objects[o] is an object in image i. If not given then
          all objects are assumed to belong to the same image.
        - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
          the spatial layout; if not given then use predicted boxes.
        """
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)           # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        # edges specify start and end.
        edges = torch.stack([s, o], dim=1)          # Shape is (T, 2)

        if obj_to_img is None:
            obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

        obj_vecs = self.obj_embeddings(objs)
        obj_vecs_orig = obj_vecs
        pred_vecs = self.pred_embeddings(p)

        # Graph convolutional network.
        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
        # what's the difference between gconv and gconv_net
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        boxes_pred = self.box_net(obj_vecs) 

        masks_pred = None
        if self.mask_net is not None:
            mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))
            masks_pred = mask_scores.squeeze(1).sigmoid()

        s_boxes, o_boxes = boxes_pred[s], boxes_pred[o]
        s_vecs, o_vecs = obj_vecs_orig[s], obj_vecs_orig[o]
        rel_aux_input = torch.cat([s_boxes, o_boxes, s_vecs, o_vecs], dim=1)
        rel_scores = self.rel_aux_net(rel_aux_input)

        H, W = self.image_size
        layout_boxes = boxes_pred if boxes_gt is None else boxes_gt


        # generate scene layout from bounding box and masks.
        if masks_pred is None:
            layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
        else:
            layout_masks = masks_pred if masks_gt is None else masks_gt
            layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks,
                                     obj_to_img, H, W)
            
        # TODO:??? what is  layout_noise_dim 
        if self.layout_noise_dim > 0: 
            N, C, H, W = layout.size()
            noise_shape = (N, self.layout_noise_dim, H, W)
            layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                       device=layout.device)
            layout = torch.cat([layout, layout_noise], dim=1)
        # TODO: layout is bbox or segmentation mask ??
        img = self.refinement_net(layout)
        return img, boxes_pred, masks_pred, rel_scores 