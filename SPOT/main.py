from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import numpy as np
import models.dcgan as dcgan
import models.mlp as mlp

import mnistm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='photo-monet', help='mnist | photo-monet')
parser.add_argument('--dataroot', default = '../data/monet2photo', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=0, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--reg', default = 5e-4, help='Reg coefficient')
parser.add_argument('--use_contour', action='store_true', help='use sobel filter as cost function')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#  
if opt.dataset == 'mnist':
    os.makedirs('data/mnist', exist_ok=True)
    dataloader_x = torch.utils.data.DataLoader(
        dset.MNIST('data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(opt.imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=opt.batchSize, shuffle=True)
    
    os.makedirs('data/mnistm', exist_ok=True)
    dataloader_y = torch.utils.data.DataLoader(
        mnistm.MNISTM('data/mnistm', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(opt.imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=opt.batchSize, shuffle=True)
#
elif opt.dataset == 'photo-monet':
    dataset1 = dset.ImageFolder(root=opt.dataroot+'/train_monet',
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    dataset2 = dset.ImageFolder(root=opt.dataroot+'/train_photo',
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))

    assert dataset1
    assert dataset2
    dataloader_x = torch.utils.data.DataLoader(dataset1, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))
    dataloader_y = torch.utils.data.DataLoader(dataset2, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))



ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def c_function(x, y):
    return torch.norm(x-y)

class C_contour(nn.Module):
    def __init__(self):
        super(C_contour, self).__init__()
        self.L1 = nn.Conv2d(opt.nc, 1, 3, bias=False)
        C1 = [[-1,0,1],[-2,0,2], [-1,0,1]]
        C1 = Tensor(C1)
        list(self.L1.parameters())[0].data[0,:,:,:] = C1.unsqueeze(0)
        list(self.L1.parameters())[0].requires_grad = False
        
        self.L2 = nn.Conv2d(opt.nc, 1, 3, bias=False)
        C2 = [[1,2,1],[0,0,0], [-1,-2,-1]]
        C2 = Tensor(C2)
        list(self.L2.parameters())[0].data[0,:,:,:] = C2.unsqueeze(0)
        list(self.L2.parameters())[0].requires_grad = False
        
    def forward(self, x, y):
        imgx1 = self.L1(x)
        imgx2 = self.L2(x)
        
        imgy1 = self.L1(y)
        imgy2 = self.L2(y)

        return torch.norm(imgx1-imgy1)+torch.norm(imgx2-imgy2)


netGx = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
netGy = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

netGx.apply(weights_init)
netGy.apply(weights_init)
if opt.netG != '': # load pretrained model if needed
    netGx.load_state_dict(torch.load(opt.netG))
    netGy.load_state_dict(torch.load(opt.netG))
print(netGx)


netDx = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
netDx.apply(weights_init)

netDy = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
netDy.apply(weights_init)

##########################################################################
#for p, q in zip(netGx.parameters(),netGy.parameters()):
#    p.data = q.data
#    
#for p, q in zip(netDx.parameters(),netDy.parameters()):
#    p.data = q.data

if opt.netD != '':
    netDx.load_state_dict(torch.load(opt.netD))
print(netDx)

if opt.netD != '':
    netDy.load_state_dict(torch.load(opt.netD))


input_x = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_y = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1
c = C_contour()

if opt.cuda:
    netDx.cuda()
    netGx.cuda()
    netDy.cuda()
    netGy.cuda()
    c.cuda()
    input_x = input_x.cuda()
    input_y = input_y.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
   
# setup optimizer
params_D = list(netDx.parameters()) + list(netDy.parameters())
params_G = list(netGx.parameters()) + list(netGy.parameters())
if opt.adam:
    optimizerD = optim.Adam(params_D, lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(params_G, lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    
    optimizerD = optim.RMSprop(params_D, lr = opt.lrD)
    optimizerG = optim.RMSprop(params_G, lr = opt.lrG)

gen_iterations = 0
for epoch in range(opt.niter):
    data_iter_x = iter(dataloader_x)
    data_iter_y = iter(dataloader_y)
    
    i = 0
    while i < len(dataloader_x)-1:
        ############################
        # (1) Update D network
        ###########################
        for p in netDx.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update
        for p in netDy.parameters(): # reset requires_grad
            p.requires_grad = True
        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 10
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < len(dataloader_x)-1:
            j += 1

#             clamp parameters to a cube
            for p in netDx.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
            for p in netDy.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data_x = data_iter_x.next()
            data_y = data_iter_y.next()
            
            if len(data_x[0])!=len(data_y[0]):
                continue              
            
            i += 1

            # train with real
            real_cpu_x, _ = data_x
            real_cpu_y, _ = data_y
           
            real_cpu_x = real_cpu_x.expand_as(real_cpu_y)

            netDx.zero_grad()
            netDy.zero_grad()
            batch_size = real_cpu_x.size(0)

            if opt.cuda:
                real_cpu_x = real_cpu_x.cuda()
                real_cpu_y = real_cpu_y.cuda()
            input_x.resize_as_(real_cpu_x).copy_(real_cpu_x)
            input_y.resize_as_(real_cpu_y).copy_(real_cpu_y)
            inputv_x = Variable(input_x)
            inputv_y = Variable(input_y)

            errD_real_x = netDx(inputv_x)
            errD_real_x.backward(one)
            
            errD_real_y = netDy(inputv_y)
            errD_real_y.backward(one)

            # train with fake
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise, volatile = True) # totally freeze netG
            fake_x = Variable(netGx(noisev).data)
            fake_y = Variable(netGy(noisev).data)
            inputv_x = fake_x
            inputv_y = fake_y
            netDx_input = netDx(inputv_x)
            netDy_input = netDy(inputv_y)
            errD_fake = netDx_input + netDy_input 

            errD_fake.backward(mone)
            errD = errD_real_x + errD_real_y - errD_fake
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netDx.parameters():
            p.requires_grad = False # to avoid computation
        for p in netDy.parameters():
            p.requires_grad = False # to avoid computation
        netGx.zero_grad()
        netGy.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake_x = netGx(noisev)
        fake_y = netGy(noisev)

        errG = netDx(fake_x) + netDy(fake_y) 
        if opt.use_contour:
            errG = errG +opt.reg* c(fake_x, fake_y)
        else:
            errG = errG +opt.reg* c_function(fake_x, fake_y)

        errG.backward(one)
        optimizerG.step()
        gen_iterations += 1
        

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.niter, i, len(dataloader_x), gen_iterations,
            errD.data[0], errG.data[0], errD_real_x.data[0], errD_fake.data[0]))
        if gen_iterations % 500 == 0:
            real_cpu_x = real_cpu_x.mul(0.5).add(0.5)
            vutils.save_image(real_cpu_x, '{0}/real_samplesx.png'.format(opt.experiment))
            fake_x = netGx(Variable(fixed_noise, volatile=True))
            fake_x.data = fake_x.data.mul(0.5).add(0.5)
            vutils.save_image(fake_x.data, '{0}/fake_samplesx_{1}.png'.format(opt.experiment, gen_iterations))
            
            real_cpu_y = real_cpu_y.mul(0.5).add(0.5)
            vutils.save_image(real_cpu_y, '{0}/real_samplesy.png'.format(opt.experiment))
            fake_y = netGy(Variable(fixed_noise, volatile=True))
            fake_y.data = fake_y.data.mul(0.5).add(0.5)
            vutils.save_image(fake_y.data, '{0}/fake_samplesy_{1}.png'.format(opt.experiment, gen_iterations))

    # do checkpointing
    torch.save(netGx.state_dict(), '{0}/netGx_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netGy.state_dict(), '{0}/netGy_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))

    
