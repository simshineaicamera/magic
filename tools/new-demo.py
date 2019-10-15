import glob
from test import *
import time
import numpy as np
import torch.multiprocessing as mp
import queue as Queue
from multiprocessing import Pool
import itertools
from multiprocessing.dummy import Pool as ThreadPool 
from pathos.multiprocessing import ProcessingPool  
# import model as modellib  
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/cup', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()




if __name__ == '__main__':
    # Setup device
    model="fast"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel(detection_speed=model)
    
    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image fil

    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]
    
    # cap = cv2.VideoCapture("/home/ubuntu/Desktop/Videos/nascar_01.mp4")

    cap = cv2.VideoCapture("/home/inomjon/Projects/Magic_Project/train/data/Images_xmls/videos/cup1.mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('/home/inomjon/Projects/Magic_Project/train/data/Images_xmls/videos/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

    if(cap.isOpened() == False):
        print("Unable to open")
        exit()
    ret, frame = cap.read()
    
    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        ROIs = cv2.selectROIs('SiamMask', frame, False, False)
    except:
        print("exit")
        exit()
    targets = []
    for i in  ROIs:
        x,y,w,h = i
    f = 0
    toc = 0
    while(cap.isOpened()):
  # Capture frame-by-frame
      ret, frame = cap.read()
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      if ret == True:
        tic = cv2.getTickCount()

        #here-----------------------
        # if f == 0:  # init
        #     count =0
        #     for i in ROIs:
        #         x,y,w,h= i
        #         target_pos = np.array([x  + w / 2, y + h / 2])
        #         target_sz = np.array([w, h])
        #         s ={"target_pos":target_pos,"target_sz":target_sz,"x":x,"y":y,"w":w,"h":h}
        #         targets.append(s)
        
        #     for i in targets:
        #         print(i["target_pos"])
        #         print(i["target_sz"])
                
        #     # state = siamese_init(frame,tar  siammask, cfg['hp'], device=device,targets=targets)  # init tracker
        #     state = siamese_init(frame, siammask, cfg['hp'], device=device,targets=targets,detector=detector)  # init tracker
        #     # state1 = siamese_init(frame, target_pos1, target_sz1, siammask, cfg['hp'], device=device)  # init tracker

        # elif f > 0:  # tracking

        #     state = siamese_track(state, frame)
# to here----------------------------
            # pool = ProcessingPool(nodes=1)
            # state =pool.map(siamese_track, state, frame)
            # results = pool.map(multi_run_wrapper,[(1,2),(2,3),(3,4)])
            # state = pool.starmap(siamese_track,zip(state, frame))
            # [state, frame, mask_enable=True, refine_enable=True, device=device)]
            # my_queue = Queue.Queue()
            # processes=[]
            # for rank in range(num_processes):
            #     # state = siamese_track(state, frame, mask_enable=True, refine_enable=True, device=device)  # track
            #     refine_enable=True
            #     mask_enable=True
            #     device=device

            #     p = mp.Process(target=siamese_track,args=(state, frame, mask_enable,refine_enable,my_queue, ))
            #     p.start()
            #     processes.append(p)
            # for p in processes:
            #     p.join()
            #     state = my_queue.get()
            #     print(state)
                
                #here ----------------------
            
            # for i,target in enumerate(state["targets"]):
                
            #     location = target['ploygon'].flatten()
            #     mask = target['mask'] > state['p'].seg_thr
            #     masks = (mask > 0) * 255     
            #     masks = masks.astype(np.uint8)
            #     frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
            # # frame[:, :, 2] = (mask1 > 0) * 255 + (mask1 == 0) * frame[:, :, 2]
            #     cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
           #-------
        cv2.imshow('SiamMask', frame)
          
        print (time.ctime()) 
         
        print("23")
        f= f+1        
        toc += cv2.getTickCount() - tic
        toc /= cv2.getTickFrequency()
        fps = f / toc
        
        # Display the resulting frame
        # cv2.imshow('Frame',frame)
     
        # Press Q on keyboard to  exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #   break
     
      # Break the loop
      else: 
        break

  #   out.release()
    # When everything done, release the video capture object
    cap.release()
     
    # Closes all the frames
    cv2.destroyAllWindows()
