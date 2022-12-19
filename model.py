import argparse
import os
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from utils import ROOT_PATH

MODEL_NAMES = ['vit_base_patch16_224', 
               'deit_base_distilled_patch16_224', 
               'levit_256', 
               'pit_b_224', 
               'cait_s24_224', 
               'convit_base', 
               'tnt_s_patch16_224', 
               'visformer_small',
               'inception_v3']

CORR_CKPTS = ['jx_vit_base_p16_224-4ee7a4dc.pth',
              'deit_base_distilled_patch16_224-df68dfff.pth',
              'LeViT-256-13b5763e.pth',
              'pit_b_820.pth',
              'S24_224.pth',
              'convit_base.pth',
              'tnt_s_patch16_224.pth.tar',
              'visformer_small-839e1f5b.pth']

CNN_MODEL_NAMES = ['inception_v3',
                   'inception_v4',
                   'inception_resnet_v2',
                   'resnetv2_152x2_bit_teacher',
                   'res2net101_26w_4s',
                   'adv_inception_v3',
                   'ens_adv_inception_resnet_v2']

def get_model(model_name):
        if model_name in MODEL_NAMES or model_name in CNN_MODEL_NAMES:
                model = create_model(
                        model_name,
                        pretrained=True,
                        num_classes=1000,
                        in_chans=3,
                        global_pool=None,
                        scriptable=False)
#         print ('Loading Model.')
        return model