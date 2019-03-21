###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
from torchvision import transforms, datasets

from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, BatchNorm2d
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from option import Options
_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]


def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
  """Converts the predicted label for evaluation.

  There are cases where the training labels are not equal to the evaluation
  labels. This function is used to perform the conversion so that we could
  evaluate the results on the evaluation server.

  Args:
    prediction: Semantic segmentation prediction.
    train_id_to_eval_id: A list mapping from train id to evaluation id.

  Returns:
    Semantic segmentation prediction whose labels have been changed.
  """
  converted_prediction = prediction.copy()
  for train_id, eval_id in enumerate(train_id_to_eval_id):
    converted_prediction[prediction == train_id] = eval_id

  return converted_prediction



if __name__ == "__main__":
    args = Options().parse()

    imroot = "../datasets/demo_images"
    imlst="..  /datasets/demo_images/images.txt"

    outdir = '%s/danet_vis' % (imroot)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outdir_eval = '%s/danet_eval' % (imroot)
    if not os.path.exists(outdir_eval):
        os.makedirs(outdir_eval)


    model = get_segmentation_model(args.model, dataset=args.dataset,
                                   backbone=args.backbone, aux=args.aux,
                                   se_loss=args.se_loss, norm_layer=BatchNorm2d,
                                   base_size=args.base_size, crop_size=args.crop_size,
                                   multi_grid=args.multi_grid, multi_dilation=args.multi_dilation)
    if args.resume_dir is None or not os.path.isdir(args.resume_dir):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_dir))
    for resume_file in os.listdir(args.resume_dir):
        if os.path.splitext(resume_file)[1] == '.tar':
            args.resume = os.path.join(args.resume_dir, resume_file)

    # resuming checkpoint
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    # strict=False, so that it is compatible with old pytorch saved models
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()
    # data transforms
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    imlist_file= open(imlst)
    imfiles = imlist_file.readlines()
    imlist_file.close()

    #attention hw x hw: 64x128x64x128
    locs = [32,64]
    vloc = 31*128+63
    origin_locs = [32*8,64*8]

    for imname in imfiles:
        imname = imname.strip()
        im_path = os.path.join(imroot, imname)
        img = Image.open(im_path).convert('RGB')
        # img = img.resize((1024,512),Image.BILINEAR)
        img_tensor = input_transform(img).unsqueeze(0).cuda()


        outputs, attention = model(img_tensor)

        # outputs = torch.nn.functional.interpolate(outputs, size=[1024,2048], scale_factor=None, mode='bilinear', align_corners=True)

        predicts = torch.max(outputs[0], 1)[1].cpu().numpy()

        mask = utils.get_mask_pallete(predicts, args.dataset)
        outname = im_path.split('/')[-1].split('.')[0] + '.png'
        mask.save(os.path.join(outdir, outname))
        # eval mask with labelid
        # eval_mask = _convert_train_id_to_eval_id(predicts, _CITYSCAPES_TRAIN_ID_TO_EVAL_ID)
        # eval_mask = Image.fromarray(eval_mask.squeeze().astype('uint8'))
        # outname = im_path.split('/')[-1].split('.')[0] + '.png'
        # eval_mask.save(os.path.join(outdir_eval, outname))
        attention_np = (attention/torch.max(attention,1)[0]).squeeze().detach().cpu().numpy()
        attention_map = (attention_np*255).astype('uint8')
        # attention_p=attention_map[vloc,:].reshape((64,128))
        # eval_mask = Image.fromarray(attention_p)
        eval_mask = Image.fromarray(attention_map)
        outname = im_path.split('/')[-1].split('.')[0] + '.png'
        eval_mask.save(os.path.join(outdir_eval, outname))







