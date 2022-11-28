import argparse
import os

import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
from torchvision import models

from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation
from resnet_cbam import resnet50_cbam, resnet34_cbam


def get_net(net_name, weight_path=None):
    """
    Get model based on network name
    :param net_name: Network Name
    :param weight_path: Training weight path
    :return:
    """
    if net_name in ['resnet_34']:
        net = models.resnet34(pretrained=False, num_classes=1)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif net_name == 'resnet_50':
        net = models.resnet50(pretrained=False, num_classes=1)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif net_name in ['resnet_cbam_50']:
        net = resnet50_cbam(pretrained=False, num_classes=1)
    elif net_name in ['resnet_cbam_34']:
        net = resnet34_cbam(pretrained=False, num_classes=1).eval()
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    # Contains the weight parameter of the specified path
    if weight_path is not None:
        net.load_state_dict(torch.load(weight_path))
    return net


def get_last_conv_name(net):
    """
    Get the name of the last convolutional layer of the network
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()

    # Normalization
    means = np.array(0.0601)
    stds = np.array(0.1734)
    image -= means
    image /= stds
    image = image[:, :, np.newaxis]
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # Add batch dimension

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    CAM
    :param image: [H,W,C],Original image
    :param mask: [H,W],0~1
    :return: tuple(cam,heatmap)
    """
    # mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # Merge heatmap to original image
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    Standardized images
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation Gradient of the input image
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)


def main(args):
    # input
    img = io.imread(args.image_path)
    img = np.float32(img) / 255
    inputs = prepare_input(img)
    # input image
    image_dict = {}
    # net
    net = get_net(args.network, args.weight_path)
    # Grad-CAM
    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    print(layer_name)
    grad_cam = GradCAM(net, layer_name)
    mask = grad_cam(inputs, args.class_id)  # cam mask
    image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
    grad_cam.remove_handlers()
    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
    mask_plus_plus = grad_cam_plus_plus(inputs, args.class_id)  # cam mask
    image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    gbp = GuidedBackPropagation(net)
    inputs.grad.zero_()  # Gradient zeroing
    grad = gbp(inputs)

    gb = gen_gb(grad)
    image_dict['gb'] = norm_image(gb)
    # Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict['cam_gb'] = norm_image(cam_gb)

    save_image(image_dict, os.path.basename(args.image_path), args.network, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet_cbam_50',
                        help='ImageNet classification network')
    parser.add_argument('--image-path', type=str, default='./input/3.png',
                        help='input image path')
    parser.add_argument('--weight-path', type=str, default='./res.pt',
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default='layer4.2.conv2',
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='output directory to save results')
    arguments = parser.parse_args()

    main(arguments)
