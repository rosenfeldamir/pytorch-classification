'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from os.path import expanduser
homeDir = expanduser('~')
import sys
sys.path.append(os.path.join(homeDir, 'YellowFin_Pytorch/tuner_utils/'))
from yellowfin import YFOptimizer

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from torch.optim.lr_scheduler import CosineAnnealingLR


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names+=['squeezenet1_1']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')

parser.add_argument('--sgdr', type=int, default=-1,
                        help='use cosine learning rate for specific number of epochs (just one cycle)')

parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--dim-slice', default=0, type=int,
                    help='on which dimension to split the conv. layer filters? (default 0)')

parser.add_argument('--lateral-inhibition', default='none', type=str,
                    help='type of lateral inhibition to apply, (default "none", means do nothing)')
parser.add_argument('--learn-inhibition', type=str2bool,default=False,
                    help='whether to learn the lateral inhibition layers or keep them fixed. ')
parser.add_argument('--half', type=str2bool,default=False,
                    help='whether to cast everything to 16bit by using half. ')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--learn-bn', type=str2bool,default=True,
                    help='whether to learn batchnorm layers or not (useful for fine-tuning)')

parser.add_argument('--print-params-and-exit', type=str2bool,default=False,
                    help='just print number of parameters to file')

parser.add_argument('--only-last', type=str2bool,default=False,
                    help='whether to learn only the last layer or not.')


parser.add_argument('--subsample', type=float,default=1.0,
                    help='subsampling ratio for quick evaluation of training methods')
parser.add_argument('--test-subsample', type=float,default=1.0,
                    help='subsampling (test) ratio for quick evaluation of training methods')
#Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('-r', '--retrain-layer', dest='retrain_layer',type=str, default='none', help='which layer to retrain')
parser.add_argument('-f', '--only-layer', dest='only_layer',type=str, default='none', help='which layer to train (only this layer!)')
parser.add_argument('-o', '--optimizer', type=str, default='sgd', help='optimizer')

parser.add_argument('-p', '--part', type=float, default=-1, help='part(fraction) of filters to learn at each layer.')
parser.add_argument('--load-fixed-path', type=str, default='', help='path of model from which to load fixed part (for ensembling)')


parser.add_argument('--zero-fixed-part', type=str2bool,default=False,
                    help='retain fixed convolutional filters or zero them out (effectively reducing number of filters')
parser.add_argument('--class-subset', default='_',
                    help='which subset of classes to train on (delimited with _, for example 3_5). if not specified, train on all classes.',
                    type=str)

#parser.add_argument('--req-perf-after-10-epochs', default=-1,type=int,
#                    help='stop after 10 epochs if this minimal performance is not obtained, -1 to ignore this option(default)',

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

assert not ('partial' in args.arch and args.part==-1),'partial architecture and part of -1 mutually exclusive.\nEither choose part > 0 or non-partial architecture.'

print('PART:',args.part)

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

if args.sgdr > 0:
    args.epochs = args.sgdr

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def split_model(model_full_dict, model_partial):
    ''' Splits a fully parametrized model's convolutional parameters into
        those of one with partial convolutions.
    '''
    partial_dict = model_partial.state_dict()
    for a, W in model_full_dict.items():
        if 'conv' in a:  # copy over the split convolutional layers.
            # find the correponding parameter in the partial model.
            a_partial_fixed = a.replace('weight', 'W_fixed')
            a_partial_learn = a.replace('weight', 'W_learn')
            W_fixed = partial_dict[a_partial_fixed]
            W_learn = partial_dict[a_partial_learn]
            W_fixed = W[:len(W_fixed)]
            W_learn = W[len(W_fixed):]
            partial_dict[a_partial_fixed] = W_fixed
            partial_dict[a_partial_learn] = W_learn
        else:
            partial_dict[a] = W
        model_partial.load_state_dict(partial_dict)


def reinit_model_layer(model, retrain_layer, initial_dict):
    
    # copy over from the initial dict's layer to the existing model
    # for retraining

    assert retrain_layer in ['conv1','block1','block2','block3','fc'],'must choose to retrain an existing layer'
    print('re-initializing model with layer:',retrain_layer)
    sd = model.state_dict()
    foundTheLayer=False
    for k in sd.keys():
        if retrain_layer in k:
            if retrain_layer == 'conv1' and 'block' in k:
                continue
            print ('AHA')
            sd[k] = initial_dict[k]
            foundTheLayer=True
    assert foundTheLayer,'could not find a matching layer name to reinitialize!!, given layer name was:'+retrain_layer
    model.load_state_dict(sd)
    # freeze all parameters except for the required layer.
    for p in model.parameters():
        p.requires_grad = False
    
    if type(model) is torch.nn.DataParallel:
        theLayer = getattr(model.module,retrain_layer)
    else:
        theLayer = getattr(model,retrain_layer)
    
    for p in theLayer.parameters():
        p.requires_grad = True
    return model

def only_layer(model, theLayer):
    
    # copy over from the initial dict's layer to the existing model
    # for retraining
    model_name = str(type(model.module)).lower()
    valid_layers = ['conv1','block1','block2','block3','fc','train_nothing']

    if theLayer == 'train_nothing':
        for p in model.parameters():
            p.requires_grad = False
        return model

    print('!!!!',model_name)
    if 'dense' in model_name:
        print('YES')
        valid_layers = [p.replace('block','dense') for p in valid_layers]
        theLayer = theLayer.replace('block', 'dense')

    assert theLayer in valid_layers, 'train_only: must choose to train an existing layer'


    print('freezing all layers except:',theLayer)
    
    # freeze all parameters except for the required layer.
    for p in model.parameters():
        p.requires_grad = False
    
    if type(model) is torch.nn.DataParallel:
        theLayer = getattr(model.module,theLayer)
    else:
        theLayer = getattr(model,theLayer)
    
    for p in theLayer.parameters():
        p.requires_grad = True
    return model
    
def trainableParams(model):
    return [p for p in model.parameters() if p.requires_grad]

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root=os.path.join(homeDir,args.dataset), train=True, download=True, transform=transform_train)

    train_sampler=None
    toShuffle = True
    if args.subsample < 1:
        toShuffle=False
        n = int(float(len(trainset)) * args.subsample)
        assert n > 0,'must sample a positive number of training examples.'
        train_sampler = data.sampler.SubsetRandomSampler(range(n))
        print('==>SAMPLING FIRST',n,'TRAINING IMAGES')


    if args.class_subset != '_':
        print('*'+args.class_subset+'*')
        toShuffle = False
        args.class_subset = [int(i) for i in args.class_subset.split('_')]
        indices = [i for i, p in enumerate(trainset.train_labels) if p in args.class_subset]
        train_sampler = data.sampler.SubsetRandomSampler(indices);
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=toShuffle, num_workers=args.workers,sampler=train_sampler)

    testset = dataloader(root=os.path.join(homeDir ,args.dataset), train=False, download=False, transform=transform_test)
    test_sampler=None
    if args.test_subsample < 1:
        n = int(float(len(testset)) * args.test_subsample)
        assert n > 0, 'must sample a positive number of training examples.'
        test_sampler = data.sampler.SubsetRandomSampler(range(n))
        print('==>SAMPLING FIRST', n, 'TESTTING IMAGES')

    if type(args.class_subset) is list:
        toShuffle = False
        indices = [i for i, p in enumerate(testset.test_labels) if p in args.class_subset]
        test_sampler = data.sampler.SubsetRandomSampler(indices);

    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers,sampler=test_sampler)

    # Model   
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        if 'partial' in args.arch:
            model = models.__dict__[args.arch](
	                    num_classes=num_classes,
	                    depth=args.depth,
	                    growthRate=args.growthRate,
	                    compressionRate=args.compressionRate,
	                    dropRate=args.drop,part=args.part, zero_fixed_part=args.zero_fixed_part,do_init=True,
                split_dim = args.dim_slice
	                )   
        else:
            model = models.__dict__[args.arch](
	                    num_classes=num_classes,
	                    depth=args.depth,
	                    growthRate=args.growthRate,
	                    compressionRate=args.compressionRate,
	                    dropRate=args.drop,lateral_inhibition=args.lateral_inhibition
	                )        
    elif args.arch.startswith('wrn'):
        if 'partial' in args.arch:
            print('==> initializing partial learning with p=',args.part)

            print('classes',num_classes,'depth',args.depth,'widen',args.widen_factor,'drop',args.drop,'part:',args.part)

            model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop, part=args.part, zero_fixed_part=args.zero_fixed_part
            )
        else:
            model = models.__dict__[args.arch](
                        num_classes=num_classes,
                        depth=args.depth,
                        widen_factor=args.widen_factor,
                        dropRate=args.drop,lateral_inhibition=args.lateral_inhibition
                    )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        if 'partial' in args.arch:
            print ('PARTIAL!!!!',args.arch)

            #print '!!!!!!!!!!!!!!!!1'
            model = models.__dict__[args.arch](num_classes=num_classes,part=args.part,zero_fixed_part=args.zero_fixed_part,do_init=True)
        else:
            print('BOOYAH---------------!!')
            model = models.__dict__[args.arch](num_classes=num_classes)





   # hack hack
    #print('==============================', arch,'===============')
    #if 'squeeze' in args.arch:
    #    model.classifier = nn.Sequential(nn.Dropout(.5), nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1)),
    #                                     nn.ReLU())
    #    print(model)
    from copy import deepcopy
    model = torch.nn.DataParallel(model).cuda()
    if args.load_fixed_path != '':
        # load the (presumably) full model dict.
        print('ensembling - loading old dict')
        fixed_model_dict = torch.load(args.load_fixed_path)['state_dict']

        #if 'partial'

        if 'partial' in args.arch:
            print('transerring to new and splitting')
            split_model(fixed_model_dict,model)
        else: # just load the dictionary as is and continue from this point.
            model.load_state_dict(fixed_model_dict)
    if False:
        if args.load_fixed_path != '':
            print('ENSEMBLING')
            # load the fixed part of this classifier for ensembling

            fixed_model_dict = torch.load(args.load_fixed_path)['state_dict']
            model_dict = model.state_dict()

            if args.part == -1:
                print('-------------BABU-----------------')
                model.load_state_dict(fixed_model_dict)

            elif args.only_layer != 'none': # just copy everything, the layer will be reinitialized layer.
                for a,b in model_dict.items():
                    if args.only_layer not in a:
                        model_dict[a] = deepcopy(fixed_model_dict[a])
                model.load_state_dict(model_dict)
            else:
                for a, b in model_dict.items():
                    # transfer all fixed values from loaded dictionary.
                    if args.part < .5: # re-train learned part.
                        if 'fixed' in a:
                            model_dict[a] = deepcopy(fixed_model_dict[a])
                    else: # re-train what was at first the random part :-)
                        if 'learn' in a:
                            model_dict[a] = deepcopy(fixed_model_dict[a])
                model.load_state_dict(model_dict)

                if args.part >= .5: # switch training between fixed / learned parts.
                    print('HAHA, SWITCHING FIXED AND LEARNING')
                    for a,b in model.module.named_parameters():
                        if 'learn' in a:
                            b.requires_grad = False
                        else:
                            b.requires_grad = True
                        # otherwise, keep it as it is.
    
    assert not (args.retrain_layer != 'none' and args.only_layer != 'none'),'retrain-layer and only-layer options are mutually exclusive'
    
    if args.retrain_layer != 'none':        
        initial_dict = deepcopy(model.state_dict())
        
    
    cudnn.benchmark = True

    #model = models.squeezenet1_1()
    #

    criterion = nn.CrossEntropyLoss()
    opt_ = args.optimizer.lower()
    if args.only_layer != 'none':
        model = only_layer(model,args.only_layer)


    # apply the learn-bn.
    for m1,m2 in model.named_modules():
        if 'bn' in m1:
            for p in m2.parameters():
                p.requires_grad = args.learn_bn
    if args.learn_inhibition:
        for p in model.module.parameters():
            p.requires_grad=True
    
    params = trainableParams(model)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000.0))
    if opt_ == 'sgd':
        print('optimizer.... - sgd')
        optimizer = optim.SGD(params , lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_ == 'adam':
        optimizer = optim.Adam(params)
    elif opt_ == 'yf':
        print('USING YF OPTIMIZER')
        optimizer = YFOptimizer(
            params, lr=args.lr, mu=0.0, weight_decay=args.weight_decay, clip_thresh=2.0, curv_win_width=20)
        optimizer._sparsity_debias = False
    else:
        raise Exception('unsupported optimizer type',opt_)

    nParamsPath = os.path.join(args.checkpoint, 'n_params.txt')
    with open(nParamsPath, 'w') as f:
        s1 = 'active_params {} \n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad))
        f.write(s1)
        s2 = 'active_params {} \n'.format(sum(p.numel() for p in model.parameters()))
        f.write(s2)
    if args.print_params_and_exit:
        exit()

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoinxt..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        #args.checkpoint = os.path.dirname(args.checkpoint)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        #start_epoch = checkpoint['epoch']
        start_epoch = args.start_epoch
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
                
        if args.retrain_layer!='none':            
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=False)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
            params = trainableParams(model)
            print('number of trainable params:',len(list(params)))
            model = reinit_model_layer(model,args.retrain_layer,initial_dict)
            params = trainableParams(model)
            print('number of trainable params:',len(list(params)))
            optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=False) # Was True
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
            
    else:
        
            
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val

    #scheduler = CosineAnnealingLR( optimizer, T_max=args.epochs)#  eta_min = 1e-9, last_epoch=args.epochs)
    if args.half:
        model = model.half()

    for epoch in range(start_epoch, args.epochs):
        if args.sgdr > 0:
            #raise Exception('currently not supporting sgdr')
            scheduler.step()
        else:
            adjust_learning_rate(optimizer, epoch)
        if type(optimizer) is YFOptimizer:
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.get_lr_factor()))  # state['lr']))
        else:
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))# state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        #if req_perf_after_10_epochs > -1

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if epoch % 5 == 0: # save each 5 epochs anyway
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint,filename='checkpoint.pth.tar_'+str(epoch).zfill(4))

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    #if args.learn_bn:
   # 	model.train()
    model.train()
    #else:
   # 	model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        if args.half:
            inputs=inputs.half()
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
                
        
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        if args.half:
            inputs=inputs.half()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        if type(optimizer) is YFOptimizer:
            optimizer.set_lr_factor(optimizer.get_lr_factor() * args.gamma)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
