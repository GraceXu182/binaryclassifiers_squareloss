#!/usr/bin/env python 

import argparse  
import os
import shutil
import time

import numpy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from torch.autograd import Variable

import sys
sys.path.append('..')
# print(sys.path)
from wideresnet.wideresnet import WideResNet
import binary_dataset

# used for logging to TensorBoard
#from tensorboard_logger import configure, log_value
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--arch', default='NetSimpleConv4', type=str,
                    help='architecture: wideresnet, NetSimpleConv, NetSimpleConvNoBN')
parser.add_argument('--ratio', default='0.2', type=float,
                    help='Random label permutation ratio: 0-100%')
parser.add_argument('--epochs', default= 1000, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--init-scale', default='0.1', type=str,     
                    help='scale for weight initialization')
parser.add_argument('--exp-name', default='10K_run1', type=str,     
                    help='10K, 4K or 2K--e')
parser.add_argument('--datasetargs', default='[]', type=str,      
                    help='arguments for creating a dataset')
parser.add_argument('--init-type', default='const_norm', type=str,
                    help='type of weight initialization: const_norm or const_std')
parser.add_argument('--compnorm-type', default='none', type=str,
                    help='complexity normalization: periter, reparam')              
parser.add_argument('--disableBias', default='0', type=str,
                    help='disable bias: 1 or 0')                  
parser.add_argument('--normx1', default='L2', type=str,
                    help='normalize input: L2, BNL2')        
parser.add_argument('--overwriteExpDir', default='', type=str,
                    help='over write expdir without asking question')        
parser.add_argument('--xnorm', default='', type=str,
                    help='normalize layer input')           
parser.add_argument('--gradnorm', default='', type=str,
                    help='normalize gradient')           
parser.add_argument('--backhook', default='', type=str,
                    help='backhook')            
parser.add_argument('--nlayerMore', default='1_1', type=str,
                    help='more layers in NetSimpleConv_more: n1_n2')                   
parser.add_argument('--bnpost', default='', type=str,  
                    help='postprocessing after bn')        
parser.add_argument('--decayLast', default='', type=str,  
                    help='decayLast')            
parser.add_argument('--hasbn', default='1', type=str,  
                    help='hasbn: 1 or 0')          
parser.add_argument('--hasnonlinear', default='1', type=str,     
                    help='hasnonlinear: 1 or 0')                  
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('-c1', '--class1', default=1, type=int,
                    help='binary class1 (default: -1, non-binary)')     
parser.add_argument('-c2', '--class2', default=2, type=int,
                    help='binary class2 (default: -1, non-binary)')  
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
# parser.add_argument('--init-scale', default=0.05, type=float,
#                     help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
#parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum') 
parser.add_argument('--nesterov', default='0', type=str,
                    help='nesterov momentum: 0 or 1')                
parser.add_argument('--weight-decay', '--wd', default=0.01, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=10, type=int,
                    help='total number of layers (default: 10). Can be 10, 16, 22, 28, ...')   
parser.add_argument('--widen-factor', default=4, type=int,
                    help='widen factor (default: 4)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
# parser.add_argument('--normx1', dest='normx1', action='store_true',           
#                     help='make norm of each input image 1 (default: False)') 
parser.add_argument('--bn-affine', dest='bn_affine', action='store_true',           
                    help='use affine transformation in bn (default: False)')   
parser.add_argument('--resume', default='', type=str, 
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--name', default='WideResNet-28-20', type=str,
#                     help='name of experiment')
parser.add_argument('--expDir', default='~/torch/runs/exp1', type=str,
                    help='path of experiment') 
parser.add_argument('--tensorboard',default=True,
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--loss_type', default='MSE', type=str, help='which loss to use: CE or MSE?')
parser.add_argument('--rescale_factor', default=1, type=float, help='rescale the one hot vector by how much?')           

parser.set_defaults(augment=False)
# parser.set_defaults(normx1=False)
parser.set_defaults(bn_affine=False)    

best_prec1 = 0

summary_writer = None

valid_output_ids = None

import sys

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()       
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

#import os
   
def main():
    global args, best_prec1
    global summary_writer, valid_output_ids 

    args = parser.parse_args()
    print(args)
    #import pdb
    #pdb.set_trace()
     
    if args.tensorboard:
        #configure( os.path.expanduser(args.expDir) )
        # if os.path.isdir(args.expDir):
        #     if args.overwriteExpDir == '1':   
        #         choice = True
        #     # else:
        #     #     choice = query_yes_no(args.expDir + ' exists, do you want to delete entirely the folder?')   
        #     if choice is True:
        #         import shutil
        #         shutil.rmtree(args.expDir)
        #     else:
        #         raise ValueError('starting a fresh experiment with tensorboard on an existing directory will corrupt the existing data of the folder. ')   
        summary_writer = SummaryWriter(args.expDir)
        # from tensorboard import program
        # tb = program.TensorBoard()
        # tb.configure(argv=[None, '--logdir', args.expDir])
        # url = tb.launch()    

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    print("args: ", args)
    if args.augment:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                (4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.dataset == 'cifar10' or  args.dataset == 'cifar100':
        # ordinary CIFAR10 CIFAR100
        trainds = datasets.__dict__[args.dataset.upper()](os.path.expanduser('~/.torch/data'), train=True, download=True,transform=transform_train) 
        valds   = datasets.__dict__[args.dataset.upper()](os.path.expanduser('~/.torch/data'), train=False, transform=transform_test)
        numClass =  args.dataset == 'cifar10' and 2 or 100
        print('Use target vector size:', numClass)
         
    elif args.dataset == 'gaussian':
        # gaussian dataset
        import gaussian_dataset  
        trainds = gaussian_dataset.GaussianDataset(args.datasetargs, train=True)                
        valds   = gaussian_dataset.GaussianDataset(trainds, train=False)     
        #trainds = gaussian_dataset(dataset_args, train=True )
        #valds   = gaussian_dataset(dataset_args, train=False )
        numClass = trainds.numClass
        #pass     
    else:
        raise ValueError('unknown dataset')     

    if args.class1!=-1:
        # binary classification
        # train_loader, val_loader = binary_dataset.getCIFAR(args.dataset.upper(),transform_train,transform_test,args.batch_size,args.class1,args.class2)
        # two classes (10K training, 2K validation)
        trainds = binary_dataset.getBinaryDS(trainds,args.class1,args.class2) 
        valds   = binary_dataset.getBinaryDS(valds,args.class1,args.class2)         
        if numClass == 2: 
            valid_output_ids = [0,1]
        else:
            valid_output_ids = [args.class1, args.class2] 
    else:
        valid_output_ids = numpy.arange(numClass)
        
    # random label experiment (4/8/2021)
    import torch.utils.data as data_utils    
    y = np.array([i[0][1] for i in trainds]).astype(int)
    def get_random_labels(y, k):
        # 4/14/2021 random labels based on different sampling ratios
        import random
        N = y.shape[0] # number of training data
        random.seed(0) # seed the random generator for reproducing the results
        print(k, type(k))
        random_indices = random.sample(range(N), int(k*(N)))
        sorted_random_indx = np.sort(random_indices)
        y_small = y[sorted_random_indx]
        np.random.seed(0) # seed the random generator for reproducing the results
        y_perturb = np.random.permutation(y_small)
        hh = y.copy()
        for i, index in enumerate(sorted_random_indx):
            hh[index]=y_perturb[i]
        return hh
    
    y1 = get_random_labels(y, args.ratio)
    print('percentage: ',args.ratio)
    
    train_targets = torch.LongTensor(y1)
    train_data = torch.stack([i[0][0] for i in trainds])
    train_index = torch.LongTensor([i[1] for i in trainds])       
    train_dataset = data_utils.TensorDataset(train_data, train_targets, train_index)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs) 
    #train_loader = torch.utils.data.DataLoader(trainds, batch_size=args.batch_size, shuffle=True, **kwargs) 
    val_loader = torch.utils.data.DataLoader(valds, batch_size=args.batch_size, shuffle=True, **kwargs)  
    
        
    # build neural network model
    import extarget
    if args.arch == 'wideresnet':
        model = WideResNet(args.layers, numClass,args.widen_factor, dropRate=args.droprate, init_scale=args.init_scale,init_type=args.init_type)  # std for normal initialization
    
    elif args.arch.startswith('NetSimpleConv') and args.arch != 'NetSimpleConv':
        func = getattr(extarget,args.arch)           
        model = func(8*args.widen_factor, numClass, init_scale=args.init_scale,init_type=args.init_type, has_nonlinear= int(args.hasnonlinear), has_bn=int(args.hasbn) , xnorm = args.xnorm ) # n-layer all conv                      
        # print(model)
    elif args.arch == 'NetSimpleConv':   
        model = extarget.NetSimpleConv(8*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type, has_nonlinear= int(args.hasnonlinear), has_bn=int(args.hasbn) ) # 5-layer all conv         
    # elif args.arch == 'NetSimpleConv_old':    
    #     model = extarget.NetSimpleConv_old(8*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type, has_nonlinear= int(args.hasnonlinear) , has_bn=int(args.hasbn) ) # 5-layer all conv         
    elif args.arch == 'NetSimpleConv_more':      
        model = extarget.NetSimpleConv_more(8*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type,nlayerMore=args.nlayerMore) # 5-layer all conv 
    elif args.arch == 'NetSimpleConvNOBN':
        model = extarget.NetSimpleConvNOBN(8*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type) # 5-layer all conv , no batch norm   
    elif args.arch == 'LinearNet':
        model = extarget.LinearNet(16*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type) # linear net  
    elif args.arch == 'LinearNetWithBN':
        model = extarget.LinearNetWithBN(16*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type, has_bn=int(args.hasbn)) # linear net               
    elif args.arch == 'HiddenLayer_1':
        model = extarget.HiddenLayer_1(32*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type) # linear net         
    elif args.arch == 'HiddenLayer_2':
        model = extarget.HiddenLayer_2(8*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type,has_bn=int(args.hasbn),affine=args.bn_affine) # conv
    elif args.arch == 'HiddenLayer_2_FC':
        model = extarget.HiddenLayer_2_FC(32*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type,has_bn=int(args.hasbn),affine=args.bn_affine,xnorm=args.xnorm) # fc   
    elif args.arch == 'HiddenLayer_3_FC':
        model = extarget.HiddenLayer_3_FC(32*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type,has_bn=int(args.hasbn),affine=args.bn_affine,xnorm=args.xnorm) # fc      
    elif args.arch == 'HiddenLayer_1_NOBN':
        model = extarget.HiddenLayer_1_NOBN(32*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type) # linear net         
    elif args.arch == 'NetSimpleConvNorm1x':
        model = extarget.NetSimpleConvNorm1x(8*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type) # 5-layer all conv    
    # elif args.arch == 'NetSimpleConvNOBN':
    #     model = extarget.NetSimpleConvNOBN(8*args.widen_factor,numClass, init_scale=args.init_scale,init_type=args.init_type) # 5-layer all conv, no BN

    import extarget
    extarget.init_convnet(model,init_scale=args.init_scale,init_type=args.init_type,bn_affine=args.bn_affine,bnpost=args.bnpost)              
     
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # CUDA_VISIBLE_DEVICES=0,1
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    model.numClass = numClass
    
    # optionally resume from a checkpoint
    if args.resume:
        # import os
        args.resume = os.path.expanduser(args.resume)                   
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    if args.loss_type == 'CE':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss_type == 'MSE':
        criterion = nn.MSELoss().cuda()  # changed 20210215
    elif args.loss_type == 'L1':
        criterion = None # nn.MSELoss().cuda()
    elif args.loss_type.startswith('MISC'):    
        import misc
        criterion = getattr(misc, args.loss_type)     
    else:
        print('Error : Loss should be either [CE / MSE')

    # criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, 
    #                             momentum=args.momentum, nesterov = args.nesterov,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, 
                                momentum=args.momentum, nesterov = bool(int(args.nesterov)),           
                                weight_decay=args.weight_decay)


    if args.decayLast != '':
        decay_value = float(args.decayLast.split(',')[1])   
        args.optimizer_decay_last_layer = torch.optim.SGD(iter([model.last_layer.weight]), args.lr * 0,   
                                                          momentum=args.momentum * 0, nesterov = bool(int(args.nesterov)),
                                                          weight_decay=decay_value) 
    
    if args.backhook != '':
        import misc
        func_arg = args.backhook.split('_')
        # if len(func_arg) == 1:
        #     func_arg.append('')      
        getattr(misc,func_arg[0])(model,func_arg[1])
        
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov = args.nesterov, weight_decay=args.weight_decay)      
    
    # cosine learning rate
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs) 
    m_val=[]
    for epoch in range(args.start_epoch, args.epochs):
        ############ save progress counter
        def update_counter_file(expDir,epoch,name='epoch_counter_'):
            import os, glob
            for filename in glob.glob(os.path.join(expDir, name + "*")):     
                os.remove(filename)
            epfile = open(name + str(epoch), 'w')          
            epfile.write(str(epoch))
            epfile.close()

        # update_counter_file(args.expDir,epoch,name='epoch_counter_')         
        ###############
        
        # train the model 
        
        #mm, i, K = train(train_loader, model, criterion, optimizer, scheduler, epoch)
        train(train_loader, model, criterion, optimizer, scheduler, epoch)
        #m_val.append(mm)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch) 

        if args.tensorboard:
            print('run the following line to launch tensorboard')
            tensorboard_command = 'tensorboard --logdir ' + args.expDir
            print(tensorboard_command)     
        
        #best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
        
    #np.save('margin_results/single_img_margin_init_%s_%d.npy'%(args.init_scale, K), np.array(m_val))
    print('Best accuracy: ', best_prec1)


    # summary_writer.close()

    if args.tensorboard:
        # from tensorboard import program
        # tb = program.TensorBoard()
        # tb.configure(argv=[None, '--logdir', args.expDir])
        # url = tb.launch()
        print('run the following line to launch tensorboard')
        tensorboard_command = 'tensorboard --logdir ' + args.expDir
        print(tensorboard_command)     
        # import os
        # os.system(tensorboard_command)

global current_i
current_i = 0 

def normalize_input(x):
    norm_x = torch.sqrt( torch.sum( torch.sum( torch.sum(torch.pow(x,2),3), 2), 1) )      
    x = x / norm_x.unsqueeze(1).unsqueeze(2).unsqueeze(3) 
    return x


def enumerate_lookahead_random_labels(iterable):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (True), or if it is the last value (False). 
    """
    # Get an iterator and pull the first value.
    i = 0
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    last_new = []
    last_new.append((last[0],last[1]))
    last_new.append(last[2])
    for val in it:
        # Report the *previous* value (more to come).
        yield i, last_new, True
        val_new = []
        val_new.append((val[0],val[1]))
        val_new.append(val[2])
        last_new = val_new
        i = i + 1
    # Report the last value.
    yield i, last_new, False

def enumerate_lookahead(iterable):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (True), or if it is the last value (False). 
    """
    # Get an iterator and pull the first value.
    i = 0
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield i, last, True
        last = val
        i = i + 1
    # Report the last value.
    yield i, last, False


def input_proc(model,input):       
    global args          
    if args.normx1 == 'L2':
        input = normalize_input(input)
    elif args.normx1 == 'BNL2':
        if not hasattr(model,'input_bn'):
            # import pdb; pdb.set_trace()                                  
            model.input_bn = nn.BatchNorm2d(input.size(1), affine=False)
            model = model.cuda()            
        input = model.input_bn(input)
        input = normalize_input(input)
    
    return input

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    
    #numClass = model.numClass    
    num_classes = model.numClass
    
    # if args.dataset == 'cifar10':
    #     num_classes = 10
    # elif args.dataset == 'cifar100':
    #     num_classes = 100
    # else:
    #     print('Error : dataset should be either [cifar10 / cifar100]')
    #     sys.exit(0)

    end = time.time()

    sum_ynfn = None
    sum_fn2 = None
    sum_abs_fn = None
    
    total_n = 0
    
    global summary_writer, valid_output_ids

    global current_i
    
    ynfn_buffer = []
    var_new = {}
    p = 0
    batch_img_id_record = []
    print('epoch number:', epoch)   
    for i, ((input, target), index), has_more in enumerate_lookahead_random_labels(train_loader):
        # # save the image indices in different batches
        # batch_img_index = index.detach().cpu().numpy()
        # batch_img_id_record.append(batch_img_index)
        
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        input = input_proc(model,input)  
        
        output = model(input)
        target = target - 1 # 0515--for obtaining one-hot encoding label of shape [128,2] 
        current_minibatch_size = input.size(0)
        total_n = total_n + current_minibatch_size
        # current_output_sum = torch.sum(torch.abs(output.clone().detach()), 0)
                
        last_output = output.clone()
        #log_value('output', output, 1)    
        
        if args.loss_type == 'MSE': 
            # import pdb; pdb.set_trace()                             
            device = target.get_device()
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(target.size()[0], 1), 1)            
            #mse_weights = target_final * 4  + 1
            #print(target_final) 
            
            loss = torch.mean((output - args.rescale_factor * target_final.type(torch.float)) ** 2)
            # loss = torch.mean(prod_rho*fn - yn)**2 + nu*normalized_weigths**2
        elif args.loss_type == 'L1': 
            # import pdb; pdb.set_trace()                             
            device = target.get_device()
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(target.size()[0], 1), 1)
            #mse_weights = target_final * 4  + 1
            #print(target_final)            
            loss = torch.mean( torch.abs(output - args.rescale_factor * target_final.type(torch.float)) )  

        elif args.loss_type.startswith('MISC'):
            # target_final = target
            # loss = criterion(output, target_final, model=model)
            import misc
            num_classes = output.size(1)      
            device = target.get_device()
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(target.size()[0], 1), 1)
            
            rho_ = misc.get_norm(output.data,unsqueeze=True)
            
            output.grad = (output.data/rho_ - misc.normalize_input(target_final.type(torch.float))).detach()               

            output.grad = output.grad * rho_        
            
            #loss = torch.mean( normalize_input_detach( output - assume_norm(target_final.type(torch.float), output.clone().detach()) ) ** 2  )
            loss = torch.mean( (  misc.normalize_input(output.data) -  misc.normalize_input(target_final.type(torch.float)) ) ** 2  )      
            loss = loss.detach()     
            
        else:
            target_final = target
            loss = criterion(output, target_final)
            
        if len(valid_output_ids) == 2 and  args.loss_type == 'MSE':
            import complexity
            # compute the product of Frobenius norms of weight matrices in different layers
            prod_rho, conv_values, bn_std, bn_scaling, conv_keys, bn_keys, bn_keys2 = complexity.get_complexities(model,'fro')       
            output_2 = output[:,valid_output_ids]
            target_final_2 = target_final[:,valid_output_ids]
            yn = target_final_2[:,0] - target_final_2[:,1]
            fn = (output_2[:,0] - output_2[:,1]) / prod_rho
            ynfn = yn*fn
            ynfn_buffer.append(ynfn.clone().detach().cpu())      
            ynfn = torch.sum(ynfn,0)
            fn2  = torch.sum(fn*fn,0)
            abs_fn = torch.sum(torch.abs(fn),0)
            if sum_ynfn is None:
                sum_ynfn = ynfn
            else:
                sum_ynfn = sum_ynfn + ynfn
            
            if sum_fn2 is None:
                sum_fn2 = fn2
            else:
                sum_fn2 = sum_fn2 + fn2

            if sum_abs_fn is None:        
                sum_abs_fn = abs_fn
            else:
                sum_abs_fn = sum_abs_fn + abs_fn
        else:
            ynfn = None
            fn2  = None
            abs_fn = None
  
        # compute output 
        # output = model(input)
        # loss = criterion(output, target_final)
        
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        #summary_writer.add_scalar('out_avg', torch.mean( torch.abs( output.clone().detach() ) ).detach().cpu().numpy() , i)      
        if args.tensorboard and epoch <=40:
            import complexity 
            prod_rho, conv_values, bn_std, bn_scaling, conv_keys, bn_keys, bn_keys2 = complexity.get_complexities(model,'fro')
            prod_rho_conv, _, _ = complexity.get_complexities_conv(model,'fro')             
            # conv_keys = ['conv_' + str(x) for x in numpy.arange(len(conv_values))]  
            # bn_keys   = ['bn_' + str(x) for x in numpy.arange(len(bn_std))]
            #bn_values = [a/b for a,b in zip(bn_scaling,bn_std)]      
            if args.tensorboard and epoch <=30:
                if args.tensorboard and epoch <=3:
                    summary_writer.add_scalar('prod_rho_by_iter_less', prod_rho, current_i)
                    summary_writer.add_scalar('prod_rho_conv_by_iter_less', prod_rho_conv, current_i)

                    summary_writer.add_scalar('minibatch_mean_abs_fn_by_iter_less', (abs_fn/current_minibatch_size).detach().cpu().numpy(), current_i)        
                    
                summary_writer.add_scalar('prod_rho_by_iter_more', prod_rho, current_i)
                summary_writer.add_scalar('prod_rho_conv_by_iter_more', prod_rho_conv, current_i)   
                summary_writer.add_scalar('train_acc_by_iter', prec1.cpu().numpy(), current_i)
                summary_writer.add_scalar('train_loss_by_iter', loss.clone().detach().cpu().numpy(), current_i)                                           
                if ynfn is not None:
                    summary_writer.add_scalar('minibatch_mean_ynfn_by_iter', (ynfn/current_minibatch_size).detach().cpu().numpy(), current_i)        
                    summary_writer.add_scalar('minibatch_mean_fn2_by_iter', (fn2/current_minibatch_size).detach().cpu().numpy(), current_i)  
                    summary_writer.add_scalar('minibatch_mean_abs_fn_by_iter', (abs_fn/current_minibatch_size).detach().cpu().numpy(), current_i)   
                    summary_writer.add_scalar('minibatch_mean_ynfn_div_fn2_by_iter', (ynfn/fn2).detach().cpu().numpy(), current_i)               
            
            if args.tensorboard and epoch <=5:
                ############################ expensive ##################################          
                conv_values = [x.detach().cpu().numpy() for x in conv_values ]  
                bn_std_values   = [(1/x).detach().cpu().numpy() for x in bn_std ]
                bn_scaling_values   = [x.detach().cpu().numpy() for x in bn_scaling ]            
                #dictionary = dict(zip(conv_keys + bn_keys , conv_values + bn_values))
                dictionary = dict(zip(conv_keys + bn_keys + bn_keys2 , conv_values + bn_std_values + bn_scaling_values))
                summary_writer.add_scalars('rho_k_by_iter', dictionary, current_i)   
                summary_writer.add_histogram('output_by_iter', output[:,valid_output_ids].clone().detach().cpu().numpy(), current_i)            

            current_i = current_i + 1
        
        if True: # epoch != 0: # do not learn in the first epoch            
            if args.decayLast != '':
                decay_epoch = int(args.decayLast.split(',')[0])         
                if epoch >= decay_epoch:    
                    decay_last_layer = True
                else:
                    decay_last_layer = False      
            else:
                decay_last_layer = False
                
            # if decay_last_layer: 
            #     args.optimizer_decay_last_layer.zero_grad()       
            
            optimizer.zero_grad()
            
            if  args.loss_type.startswith('MISC'): 
                output.backward(output.grad)    
            else:
                loss.backward()

            if args.gradnorm !='':            
                import misc
                func_arg = args.gradnorm.split('_')   
                getattr(misc,func_arg[0])(model,func_arg[1])

            if args.tensorboard and epoch <=5:
                import complexity
                # gradnorm        
                _, conv_grads_values, conv_keys = complexity.get_complexities_grads_conv(model,'fro') 
                conv_grads_values = [x.detach().cpu().numpy() for x in conv_grads_values ]  
                dictionary = dict(zip(conv_keys , conv_grads_values )) 
                summary_writer.add_scalars('grad_norm_k_by_iter', dictionary, current_i)                          

            if decay_last_layer:
                args.optimizer_decay_last_layer.step()           
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            import complexity              
            
            if args.compnorm_type == 'periter':                
                complexity.compnorm_periter(model,args.init_scale,bn_affine=args.bn_affine)
            elif args.compnorm_type == 'periter2':              
                import misc       
                misc.compnorm_periter2(model,args.init_scale,bn_affine=args.bn_affine)                         
            
            if args.disableBias == '1':
                import extarget      
                extarget.disable_bias(model)   
            
            
        
        # measure the elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) or (not has_more): 
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))

    # log to TensorBoard
    if args.tensorboard:
        ## save ynfn
        if len(ynfn_buffer) > 0:
            ynfn_buffer = torch.cat(ynfn_buffer)    
            import h5py
            with h5py.File(os.path.join(args.expDir,'ynfn.h5'), 'w') as hf:
                dset = hf.create_dataset("ynfn",  data=ynfn_buffer.numpy())        
                dset.attrs['epoch'] = epoch
                     
        ## visualize layers         
        #import tensorvisualization            
        #inputdata = tensorvisualization.trainloader_data(train_loader)
        #for i in range(len(inputdata)):
        #    inputdata[i] = input_proc(model,inputdata[i])                           
        #layeract = tensorvisualization.activation_extract(model,inputdata)
        # import pdb; pdb.set_trace()                           
        #tensorvisualization.add_hist(layeract,epoch,summary_writer)
        #  layeract = None   
        ##

        import complexity

        prod_rho, conv_values, bn_std, bn_scaling, conv_keys, bn_keys, bn_keys2 = complexity.get_complexities(model,'fro')

        prod_rho_conv, _, _ = complexity.get_complexities_conv(model,'fro') 

        # torch.mean(torch.sqrt(torch.sum(torch.sum(torch.sum(layeract['conv3_sub.bn']**2,3),2),1)))
        # conv_values        
        
        if len( valid_output_ids ) == 2 and  args.loss_type == 'MSE':
            sum_ynfn = sum_ynfn/total_n  
            sum_fn2 = sum_fn2/total_n
            sum_abs_fn = sum_abs_fn/total_n 
            summary_writer.add_scalar('mean_ynfn_by_epoch', sum_ynfn.detach().cpu().numpy(),  epoch)  
            summary_writer.add_scalar('mean_fn2_by_epoch',  sum_fn2.detach().cpu().numpy(),  epoch)    
            summary_writer.add_scalar('mean_ynfn_div_fn2_by_epoch', ( sum_ynfn/ sum_fn2).detach().cpu().numpy(), epoch)               
            summary_writer.add_scalar('mean_abs_fn_by_epoch',  sum_abs_fn.detach().cpu().numpy(),  epoch)                        
            # sum_output = sum_output/total_n
            last_output_2 = last_output[:,valid_output_ids]
            #target_final_2 = target_final[:,valid_output_ids]
            #yn = target_final_2[:,0] - target_final_2[:,1]
            last_fn = (last_output_2[:,0] - last_output_2[:,1]) / prod_rho
            summary_writer.add_histogram('fn_by_epoch', last_fn.detach().cpu().numpy(), epoch)
            summary_writer.add_histogram('abs_fn_by_epoch', torch.abs(last_fn).detach().cpu().numpy(), epoch)          
            
        summary_writer.add_histogram('output_by_epoch', last_output.detach().cpu().numpy(), epoch)
        
        # conv_keys = ['conv_' + str(x) for x in numpy.arange(len(conv_values))]
        # bn_keys   = ['bn_' + str(x) for x in numpy.arange(len(bn_std))]
        
        
        conv_values = [x.detach().cpu().numpy() for x in conv_values ]          
        bn_std_values   = [(1/x).detach().cpu().numpy() for x in bn_std ]
        bn_scaling_values   = [x.detach().cpu().numpy() for x in bn_scaling ]            

        dictionary = dict(zip(conv_keys + bn_keys + bn_keys2 , conv_values + bn_std_values + bn_scaling_values))     

        # bn_values = [a/b for a,b in zip(bn_scaling,bn_std)] 
        # conv_values = [x.detach().cpu().numpy() for x in conv_values ]   
        # bn_values   = [x.detach().cpu().numpy() for x in bn_values ]     
        # dictionary = dict(zip(conv_keys + bn_keys, conv_values + bn_values))
        
        summary_writer.add_scalar('prod_rho_by_epoch', prod_rho, epoch)
        summary_writer.add_scalar('prod_rho_conv_by_epoch', prod_rho_conv, epoch)             
        summary_writer.add_scalars('rho_k_by_epoch', dictionary, epoch)                            

        # gradnorm
        _, conv_grads_values, conv_keys = complexity.get_complexities_grads_conv(model,'fro')   
        conv_grads_values = [x.detach().cpu().numpy() for x in conv_grads_values ]    
        dictionary = dict(zip(conv_keys , conv_grads_values ))     
        summary_writer.add_scalars('grad_norm_k_by_epoch', dictionary, epoch)                                
        last_output = last_output[:,valid_output_ids]       
        # print(last_output.size())            
        #summary_writer.add_histogram('output', last_output.detach().cpu().numpy(), epoch) 
        #out_diff = last_output.detach().cpu().numpy()[0,:] - last_output.detach().cpu().numpy()[1,:]        
        summary_writer.add_scalar('train_loss', losses.avg, epoch)
        summary_writer.add_scalar('train_acc', top1.avg, epoch)
        
def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    num_classes = model.numClass
    # if args.dataset == 'cifar10':
    #     num_classes = 10
    # elif args.dataset == 'cifar100':
    #     num_classes = 100
    # else:
    #     print('Error : dataset should be either [cifar10 / cifar100]')
    #     sys.exit(0)

    # switch to evaluate mode
    global summary_writer, valid_output_ids    
    model.eval()
    test_ynfn = []
    end = time.time()
    for i, ((input, target), index), has_more in enumerate_lookahead(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        target = target - 1 # 0515--for obtaining one-hot encoding label of shape [128,2] 
        # if args.normx1:
        #     input = normalize_input(input)  
        # if args.normx1 == 'L2':
        #     input = normalize_input(input)
        # elif args.normx1 == 'BNL2':
        #     # if not hasattr(model,'input_bn'):
        #     #     model.input_bn = nn.BatchNorm2d(input.size(1), affine=False)                 
        #     input = model.input_bn(input)
        #     input = normalize_input(input)        
        input = input_proc(model,input)    
        
        # compute output
        with torch.no_grad():
            output = model(input)

        if args.loss_type == 'MSE':
            device = target.get_device()
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(target.size()[0], 1), 1)
            target_final = args.rescale_factor * target_final.type(torch.float)
        else:
            target_final = target

        # compute output
        output = model(input)
        #import pdb; pdb.set_trace()                              
        # loss = criterion(output, target_final)       
        if args.loss_type == 'MSE': 
            # import pdb; pdb.set_trace()                             
            device = target.get_device()
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(target.size()[0], 1), 1)
            #mse_weights = target_final * 4  + 1
            #print(target_final)            
            import complexity
            prod_rho, conv_values, bn_std, bn_scaling, conv_keys, bn_keys, bn_keys2 = complexity.get_complexities(model,'fro')       
            output_2 = output[:,valid_output_ids]
            target_final_2 = target_final[:,valid_output_ids]
            yn = target_final_2[:,0] - target_final_2[:,1]
            fn = (output_2[:,0] - output_2[:,1]) / prod_rho # gn =  prod_rho * fn
            ynfn = yn*fn
            test_ynfn.append(ynfn.clone().detach().cpu())      
            
            loss = torch.mean((output - args.rescale_factor * target_final.type(torch.float)) ** 2)

        elif args.loss_type == 'L1': 
            # import pdb; pdb.set_trace()                             
            device = target.get_device()
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(target.size()[0], 1), 1)
            #mse_weights = target_final * 4  + 1
            #print(target_final)            
            loss = torch.mean( torch.abs(output - args.rescale_factor * target_final.type(torch.float)) )  

        else:
            target_final = target
            loss = criterion(output, target_final)


        # import pdb; pdb.set_trace()              
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) or (not has_more):   
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:        
        summary_writer.add_scalar('val_loss', losses.avg, epoch)
        summary_writer.add_scalar('val_acc', top1.avg, epoch)   
        test_ynfn = torch.cat(test_ynfn)    
        import h5py
        with h5py.File(os.path.join(args.expDir,'test_ynfn.h5'), 'w') as hf:
            dset = hf.create_dataset("test_ynfn",  data=test_ynfn.numpy())
            dset.attrs['epoch'] = epoch        
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk""" 
    #directory = "~/torch/runs/%s/"%(args.name)
    directory = os.path.expanduser(args.expDir.rstrip('/') + '/')       
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')
        #'runs/%s/'%(args.name) 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # print("maxk: ", maxk)
    # print("pred: ", pred)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print("target: ", target.view(1, -1).expand_as(pred))
    # print("correct: ", correct)
    # sys.exit()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
