import argparse
import torch
import torch.nn as nn
from dataset import EYEDataset
import numpy as np
from torch.autograd import Variable
from PIL import Image
import torchvision.models as models
from model import FCN8s


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-c", "--cuda", action='store_true', default=False)
arg_parser.add_argument("-n","--nb_worker", type=int, default=4,help='# of workers')
args = arg_parser.parse_args()

vgg16 = models.vgg16(pretrained=True)  
fcn = FCN8s(vgg16)

EYE_PATH='./256fcn/'
def test(args,fcn=None):
    model_file_f = "./models/b5.e200.fcn8s.pkl"

    if model_file_f:                                                   
        fcn.load_state_dict(torch.load(model_file_f)) 
         
    
    ds_val   = EYEDataset(EYE_PATH, set='test')
    print("Loaded {} test data.".format(len(ds_val)))
    #vis = visdom.Visdom(port=8097)
    if args.cuda:
        fcn =fcn.cuda(0)
    
    test_loader = torch.utils.data.DataLoader(dataset=ds_val,
                                               batch_size=1,num_workers=args.nb_worker) #,shuffle=False

    total_loss_f = 0.
    for i, (images,gts,base_name) in enumerate(test_loader):
        print("Loaded {}ed pic.".format(base_name[0]))
        if args.cuda:
            images = Variable(images.cuda(0))
            gts    = Variable(gts.cuda(0))
        with torch.no_grad():
            outputs =fcn(images)
            #loss_f = nn.CrossEntropyLoss()(outputs,gts)
            #total_loss_f +=loss_f.item()
            #print("loss:{}".format(loss_f))
        predicted = ds_val.decode_segmap(outputs[0].cpu().data.numpy().argmax(0))
        predicted = (predicted *255.0).astype(np.uint8)
        predicted = Image.fromarray(predicted)
        pred_path = './output/'+base_name[0]+'.png'
        predicted.save(pred_path)
    #print("avg-loss:{}".format(total_loss_f/len(test_loader)))

if __name__ == '__main__':
    test(args,fcn)
