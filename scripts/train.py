import os
import json
import argparse

from sg2im.data import imagenet_deprocess_batch
from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.utils import int_tuple, bool_flag, float_tuple, str_tuple

from reflectionNN.model import ReflectionModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

 
def get_args():

    VG_DIR = os.path.expanduser("~/simsg/datasets/vg/")
    COCO_DIR = os.path.expanduser("~/coco/")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco', choices=['vg', 'coco'])

    # Dataset options common to both VG and COCO
    parser.add_argument('--image_size', default='64,64', type=int_tuple)
    parser.add_argument('--num_train_samples', default=None, type=int)
    parser.add_argument('--num_val_samples', default=1024, type=int)
    parser.add_argument('--shuffle_val', default=True, type=bool_flag)
    parser.add_argument('--loader_num_workers', default=4, type=int)
    parser.add_argument('--include_relationships',
                        default=True, type=bool_flag)

    # VG-specific options
    parser.add_argument(
        '--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
    parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
    parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
    parser.add_argument(
        '--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
    parser.add_argument('--max_objects_per_image', default=10, type=int)
    parser.add_argument('--vg_use_orphaned_objects',
                        default=True, type=bool_flag)

    # COCO-specific options
    parser.add_argument('--coco_train_image_dir',
                        default=os.path.join(COCO_DIR, '/train2017'))
    parser.add_argument('--coco_val_image_dir',
                        default=os.path.join(COCO_DIR, '/val2017'))
    parser.add_argument('--coco_train_instances_json',
                        default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
    parser.add_argument('--coco_train_stuff_json',
                        default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
    parser.add_argument('--coco_val_instances_json',
                        default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
    parser.add_argument('--coco_val_stuff_json',
                        default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
    parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
    parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
    parser.add_argument('--coco_include_other', default=False, type=bool_flag)
    parser.add_argument('--min_object_size', default=0.02, type=float)
    parser.add_argument('--min_objects_per_image', default=3, type=int)
    parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)

    # Output options
    # parser.add_argument('--print_every', default=10, type=int)
    # parser.add_argument('--timing', default=False, type=bool_flag)
    parser.add_argument('--checkpoint_every', default=10000, type=int)
    parser.add_argument('--output_dir', default=os.getcwd())
    parser.add_argument('--checkpoint_name', default='checkpoint')
    parser.add_argument('--checkpoint_start_from', default=None)
    parser.add_argument('--restore_from_checkpoint',
                        default=False, type=bool_flag)

    args = parser.parse_args()
    return args


def build_coco_dsets(args):
    dset_kwargs = {
        'image_dir': args.coco_train_image_dir,
        'instances_json': args.coco_train_instances_json,
        'stuff_json': args.coco_train_stuff_json,
        'stuff_only': args.coco_stuff_only,
        'image_size': args.image_size,
        'mask_size': args.mask_size,
        'max_samples': args.num_train_samples,
        'min_object_size': args.min_object_size,
        'min_objects_per_image': args.min_objects_per_image,
        'instance_whitelist': args.instance_whitelist,
        'stuff_whitelist': args.stuff_whitelist,
        'include_other': args.coco_include_other,
        'include_relationships': args.include_relationships,
    }
    train_dset = CocoSceneGraphDataset(**dset_kwargs)
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)
    print('Training dataset has %d images and %d objects' %
          (num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    dset_kwargs['image_dir'] = args.coco_val_image_dir
    dset_kwargs['instances_json'] = args.coco_val_instances_json
    dset_kwargs['stuff_json'] = args.coco_val_stuff_json
    dset_kwargs['max_samples'] = args.num_val_samples
    val_dset = CocoSceneGraphDataset(**dset_kwargs)

    assert train_dset.vocab == val_dset.vocab
    vocab = json.loads(json.dumps(train_dset.vocab))

    return vocab, train_dset, val_dset


def build_vg_dsets(args):
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': args.train_h5,
        'image_dir': args.vg_image_dir,
        'image_size': args.image_size,
        'max_samples': args.num_train_samples,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.vg_use_orphaned_objects,
        'include_relationships': args.include_relationships,
    }
    train_dset = VgSceneGraphDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // args.batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    dset_kwargs['h5_path'] = args.val_h5
    del dset_kwargs['max_samples']
    val_dset = VgSceneGraphDataset(**dset_kwargs)

    return vocab, train_dset, val_dset


def build_loaders(args):
    if args.dataset == 'vg':
        vocab, train_dset, val_dset = build_vg_dsets(args)
        collate_fn = vg_collate_fn
    elif args.dataset == 'coco':
        vocab, train_dset, val_dset = build_coco_dsets(args)
        collate_fn = coco_collate_fn

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
        'collate_fn': collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = args.shuffle_val
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader, val_loader


def build_model(args, vocab):
    """Build a new model or restore from checkpoint

    Args:
        args (argparse): [description]
        vocab (dict): [description]

    Returns:
        Model: model
    """    
    if args.checkpoint_start_from is not None:
        checkpoint = torch.load(args.checkpoint_start_from)
        kwargs = checkpoint['model_kwargs']
        model = ReflectionModel(**kwargs)
        raw_state_dict = checkpoint['model_state']
        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v
        model.load_state_dict(state_dict)
    else:
        kwargs = {
            # 'vocab': vocab,
            # 'image_size': args.image_size,
            # 'embedding_dim': args.embedding_dim,
            # 'gconv_dim': args.gconv_dim,
            # 'gconv_hidden_dim': args.gconv_hidden_dim,
            # 'gconv_num_layers': args.gconv_num_layers,
            # 'mlp_normalization': args.mlp_normalization,
            # 'refinement_dims': args.refinement_network_dims,
            # 'normalization': args.normalization,
            # 'activation': args.activation,
            # 'mask_size': args.mask_size,
            # 'layout_noise_dim': args.layout_noise_dim,
        }
        model = ReflectionModel(**kwargs)
    return model, kwargs


def main():
    args = get_args()
    print(args)
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor

    vocab, train_loader, val_loader = build_loaders(args)
    # TODO: build model
    model, model_kwargs = build_model(args, vocab)
    model.type(float_dtype)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)