# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# Modified by Inomjon for auto labeling images
# --------------------------------------------------------

from PIL import Image
import os
import random
import cv2
from utils.consts import *
import glob, os
from tools.test import *
from utils.label import data_generate, auto_label, stylize
from stylize import net
import torch.nn as nn
from stylize.stylize import input_transform
from stylize.stylize import style_transfer
parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='experiments/siammask_sharp/SiamMask_DAVIS.pth', type=str,
                    metavar='PATH',help='path to latest checkpoint (default: SiamMask_DAVIS)')
parser.add_argument('--config', dest='config', default='experiments/siammask_sharp/config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='data/Images_xmls/images/', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()


progress = 0

def tracking(label):

     # Setup device
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    # ---------stylize data ----
    decoder = net.decoder
    vgg = net.vgg
    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('stylize/models/decoder.pth'))
    vgg.load_state_dict(torch.load('stylize/models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)
    content_tf = input_transform(0, False)
    style_tf = input_transform(0, False)
    # --------------------------
    # Setup Model
    cfg = load_config(args)
    from experiments.siammask_sharp.custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    #siammask, device, cfg = init_tracking_model()
    siammask.eval().to(device)

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = []
    img_names = []
    for imf in img_files:
        ims.append(cv2.imread(imf))
        imf = imf.split("/")[-1]
        img_names.append(imf)
    # Select ROI
    cv2.namedWindow("AutoLabel", cv2.WINDOW_NORMAL)
    try:
        cv2.putText(ims[0], "Please select an object and press 'Enter' to continue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        init_rect = cv2.selectROI('AutoLabel', ims[0], True, False)
        x, y, w, h = init_rect
        #cv2.destroyAllWindows()
    except Exception:
        exit()
    noObject=0
    conf = 0.99
    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0 :  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            #labelWin.showWin(labelWin)
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            init_state = state

        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            score = state["score"]
                #mask = state['mask'] > state['p'].seg_thr
                #print(location)

            if score > conf:
                conf = 0.99

                if len(location)==8:
                    xlist = []
                    ylist = []
                    for jj in range(0,8, 2):
                        xlist.append(location[jj])
                        ylist.append(location[jj +1])

                    xlist = sorted(xlist)
                    ylist = sorted(ylist)
                    #print("xlist is: ", xlist)
                    #print("ylist is:", ylist)
                    xmin = xlist[0]
                    xmax = xlist[-1]
                    ymin = ylist[0]
                    ymax = ylist[-1]
                    box = [int(xmin), int(ymin), int(xmax), int(ymax)]
                    name = img_names[f]
                    if not os.path.exists(os.path.join(jpg_path, name)):
                        os.system(copy_command.format(os.path.join(src_path, name), jpg_path))
                    height = im.shape[0]
                    width = im.shape[1]
                    depth=im.shape[2]
                    auto_label(im, box, name, label, height, width, depth)

                    data_generate(im, box, name, label)
                    #stylize(name, device,vgg, decoder, content_tf, style_tf)
                    cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                    #getProgress(f)

            # print("conf", conf)
            # print("score", score)
            cv2.imshow('AutoLabel', im)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
    cv2.destroyAllWindows()

# def getProgress(value):
#     global progress
#     progress = value

# def setProgress():
#     global progress
#     if progress is None:
#         progress = 0
#     return progress
