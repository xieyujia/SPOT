from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os,sys
import numpy as np
import torch.nn.functional as F

from utils import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='mnist', help='mnist | mnistm | usps | svhn')
parser.add_argument('--target', default='mnistm', help='mnist | mnistm | usps | svhn')
parser.add_argument('--dataroot', default = '~/dataset', help='path to dataset')
parser.add_argument('--experiment', default = None, help='path to experiment')
parser.add_argument('--n_class', type = int, default = 12, help='number of classes for claasification')
parser.add_argument('--pretrain', default = None,
                                  help='path to pretrained classification network')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--niter', type=int, default=30000, help='number of epochs to train for')
parser.add_argument('--zsize', type=int, default=100, help='dimension of random noise')
parser.add_argument('--lrP', type=float, default=0.0002, help='learning rate for netP, default=0.0001')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for netG, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd. default=0.9')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay. default=0.0005')
parser.add_argument('--opt', default='adam', help='use adam or sgd')

parser.add_argument('--d_iter', type=int, default=1, help='train Discriminater for d_iter times per iteration')
parser.add_argument('--netDlossscale', type=float, default=1.0, help='netDlossscale')
parser.add_argument('--netPlossscale', type=float, default=10.0, help='netPlossscale')
parser.add_argument('--OPTlossscale', type=float, default=0.01, help='OPTlossscale')
parser.add_argument('--OPTlossscale2', type=float, default=0.001, help='OPTlossscale2')
parser.add_argument('--resample', default=True, action='store_true', help='resample data after training netD')
parser.add_argument('--gandistance', default="dcgan", help='wgan|dcgan')
parser.add_argument('--testiter', type=int, default=500, help='testiter')

parser.add_argument('--softmaxthe', type=float, default=0.95, help='OPTlossscale2')

opt = parser.parse_args()
print(opt)

def dcganloss(x):
    return (F.softplus(x)).mean()
def wganloss(x):
    return (x).mean()

import torch.nn.init as init
def xavier_weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0.1)

class C_contour(nn.Module):
    def __init__(self):
        super(C_contour, self).__init__()
        self.L1 = nn.Conv2d(3, 1, 3, bias=False)
        C1 = np.array([[-1,0,1],[-2,0,2], [-1,0,1]])
        C1 = torch.from_numpy(C1)
        list(self.L1.parameters())[0].data[0,:,:,:] = C1.unsqueeze(0)
        list(self.L1.parameters())[0].requires_grad = False

        self.L2 = nn.Conv2d(3, 1, 3, bias=False)
        C2 = np.array([[1,2,1],[0,0,0], [-1,-2,-1]])
        C2 = torch.from_numpy(C2)
        list(self.L2.parameters())[0].data[0,:,:,:] = C2.unsqueeze(0)
        list(self.L2.parameters())[0].requires_grad = False

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = torch.cat([x,x,x],1)
        if y.shape[1] == 1:
            y = torch.cat([y,y,y],1)
        imgx1 = self.L1(x)
        imgx2 = self.L2(x)

        imgy1 = self.L1(y)
        imgy2 = self.L2(y)
#       img = img.view(img.shape[0], *img_shape)
        return torch.norm(imgx1-imgy1)+torch.norm(imgx2-imgy2)

def main():
    if opt.experiment is None:
        opt.experiment = opt.source + "2" + opt.target
    opt.experiment = "digits_exp/" + opt.experiment
    os.system('mkdir {0}'.format(opt.experiment))
    stdout_backup = sys.stdout
    sys.stdout = Logger(opt.experiment +"/log.txt","w", stdout_backup)

    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    def createDataset(dataname, train):
        if dataname == "mnist":
            return dset.MNIST(root=opt.dataroot+'/mnist', train=train, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))
        if dataname == "mnistm":
            from utils import MNISTM
            return MNISTM(root=opt.dataroot+'/mnistm', mnist_root=opt.dataroot+'/mnist', train=train, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))
        if dataname == "usps":
            from utils import USPS
            return USPS(root=opt.dataroot+'/usps', train=train, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))
        if dataname == "svhn":
            return dset.SVHN(root=opt.dataroot+'/svhn', split=("train" if train else "test"), download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(28),
                                    transforms.ToTensor()
                                ]))

    dataset1 = createDataset(opt.source, True)
    dataset2_aux = createDataset(opt.target, True)
    # for testing
    dataset2 = createDataset(opt.target, False)

    dataloader_x = torch.utils.data.DataLoader(dataset1, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers), pin_memory=True, drop_last = True)
    dataloader_yaux = torch.utils.data.DataLoader(dataset2_aux, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers), pin_memory=True, drop_last = True)
    dataloader_y = torch.utils.data.DataLoader(dataset2, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers), pin_memory=True)


    import models.model_mnist_mnistm as models
    from utils import DataAttDict

    netP = models.CoDis28x28(*DataAttDict[opt.source],*DataAttDict[opt.target]).apply(xavier_weights_init).to(device)
    netD_aux = nn.Conv2d(20, 1, kernel_size=12, stride=1, padding=0).to(device)
    netD_aux2 = nn.Sequential(
                    nn.Conv2d(500, 100, kernel_size=1, stride=1, padding=0),
                    nn.PReLU(),
                    nn.Conv2d(100, 1, kernel_size=1, stride=1, padding=0)
                ).to(device)
    netG = models.CoGen28x28(*DataAttDict[opt.source],*DataAttDict[opt.target], zsize=opt.zsize).apply(xavier_weights_init).to(device)

    # setup optimizer
    if opt.opt == "adam":
        optimizerP = optim.Adam([p for p in netP.parameters() if p.requires_grad], lr=opt.lrP, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
        optimizerD_aux = optim.Adam([p for p in netD_aux.parameters() if p.requires_grad]+[p for p in netD_aux2.parameters() if p.requires_grad], lr=opt.lrP, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
        optimizerG = optim.Adam([p for p in netG.parameters() if p.requires_grad], lr=opt.lrG, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.opt == "rmsprop":
        optimizerP = optim.RMSprop([p for p in netP.parameters() if p.requires_grad], lr=opt.lrP)
        optimizerG = optim.RMSprop([p for p in netG.parameters() if p.requires_grad], lr=opt.lrG)
    elif opt.opt == "sgd":
        optimizerP = optim.SGD([p for p in netP.parameters() if p.requires_grad], lr=opt.lrP, momentum=opt.momentum, weight_decay=opt.weight_decay)
        optimizerG = optim.SGD([p for p in netG.parameters() if p.requires_grad], lr=opt.lrG, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # define loss
    criterion = nn.CrossEntropyLoss()
    opt_contour_loss = C_contour().to(device)
    if opt.gandistance == "wgan":
        ganloss = wganloss
    elif opt.gandistance == "dcgan":
        ganloss = dcganloss

    # Define test function
    def test_f(verbose = True, print_period = 100):
        # VALIDATION
        j = 0
        cum_acc = 0
        total_len = 0

        netP.eval()
        for y, y_label in dataloader_y:
            j = j+1

            y = y.to(device)
            y_label = y_label.to(device)

            # compute output
            outputs, _ = netP.pred_t(y)
            test_loss = criterion(outputs, y_label)

            pred = torch.argmax(outputs,dim=-1)
            test_acc = torch.sum(pred==y_label).item()
            cum_acc = cum_acc+test_acc
            test_acc = test_acc/len(pred)
            total_len += len(pred)
            if j%print_period==0 and verbose:
                print('Iter: [%d/%d],  Test Loss:  %.8f, Test Acc:  %.2f' % (j,len(dataloader_y),test_loss, test_acc))
        print(' Test acc for the epoch:  %.8f\n##############################################' % (cum_acc/total_len))
        return cum_acc/total_len

    def show_tsne(xr, xl, yr, yl, xfr, xfl, yfr, yfl, epoch):
        import sklearn
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=0)
        X = np.concatenate((xr, yr, xfr, yfr), axis=0)
        X_2d = tsne.fit_transform(X)
        from matplotlib import pyplot as plt
        plt.figure(figsize=(6, 5))
        colors = np.array(['r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple'])
        plt.scatter(X_2d[:opt.batchSize, 0], X_2d[:opt.batchSize, 1], c=colors[xl], marker="o", label=["source"])
        for i in range(opt.batchSize):
            plt.text(X_2d[opt.batchSize+i, 0], X_2d[opt.batchSize+i, 1], str(yl[i]), color=colors[yl[i]], label="target")
        plt.scatter(X_2d[opt.batchSize:opt.batchSize*2, 0], X_2d[opt.batchSize:opt.batchSize*2, 1], c=colors[yl], marker="*", label=["target"])
        plt.scatter(X_2d[opt.batchSize*2:opt.batchSize*3, 0], X_2d[opt.batchSize*2:opt.batchSize*3, 1], marker="_", c=colors[xfl], label="source fake")
        plt.scatter(X_2d[opt.batchSize*3:opt.batchSize*4, 0], X_2d[opt.batchSize*3:opt.batchSize*4, 1], marker="+", c=colors[yfl], label="target fake")
        plt.legend()
        plt.savefig(opt.experiment +'/tsne_%05d.pdf'%(epoch), bbox_inches='tight',format="pdf", dpi = 300)
        plt.close()


    def adjust_learning_rate(optimizer, decay):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay

    best_test_acc = 0

    fixed_noise = x_noise = torch.randn(opt.batchSize, opt.zsize).to(device)
    fixed_x = None
    fixed_y = None

    from utils import InfIter
    yauxiter = InfIter(dataloader_yaux)
    xiter = InfIter(dataloader_x)
    for i in range(opt.niter):
        #if (i+1) % 5000 == 0:
        #    opt.OPTlossscale *= 0.1
        #    opt.netPlossscale *= 0.1
        netP.train()
        netD_aux.train()
        netD_aux2.train()
        netG.train()
        ###### Solver inner
        for d_iter in range(opt.d_iter):
            netP.zero_grad()
            netD_aux.zero_grad()
            netD_aux2.zero_grad()
            # Load data
            y, y_label = next(yauxiter)
            x, x_label = next(xiter)
            z_noise = torch.randn(opt.batchSize, opt.zsize).to(device)

            x = x.to(device)
            y = y.to(device)
            x_label = x_label.to(device)

            # GAN training
            x_fake, y_fake = netG(z_noise)
            x_out_d,x_out_daux,x_rep, y_out_d,y_out_daux,y_rep = netP(x,y)
            x_out_d_fake,_,x_rep_fake,y_out_d_fake,_,y_rep_fake = netP(x_fake.detach(), y_fake.detach())

            errD_xy = ganloss(netD_aux(x_out_daux.detach())) * opt.netDlossscale
            errD_xy += ganloss(-netD_aux(y_out_daux.detach())) * opt.netDlossscale
            errD_xy += ganloss(netD_aux2(x_rep.detach())) * opt.netDlossscale
            errD_xy += ganloss(-netD_aux2(y_rep.detach())) * opt.netDlossscale
            errD_xy.backward()
            optimizerD_aux.step()

            errD_xy = ganloss(-netD_aux(x_out_daux)) * opt.netDlossscale
            errD_xy += ganloss(netD_aux(y_out_daux)) * opt.netDlossscale
            errD_xy += ganloss(-netD_aux2(x_rep.detach())) * opt.netDlossscale
            errD_xy += ganloss(netD_aux2(y_rep.detach())) * opt.netDlossscale


            errD_x_real = ganloss(x_out_d) * opt.netDlossscale
            errD_y_real = ganloss(y_out_d) * opt.netDlossscale
            errD_x_fake = ganloss(-x_out_d_fake) * opt.netDlossscale
            errD_y_fake = ganloss(-y_out_d_fake) * opt.netDlossscale
            D_x_real = x_out_d.mean().item()
            D_y_real = y_out_d.mean().item()
            D_x_fake = x_out_d_fake.mean().item()
            D_y_fake = y_out_d_fake.mean().item()

            x_out = netP.pred_fromrep(x_rep, target = x_label)
            netP.eval()
            x_out_fake = netP.pred_fromrep(x_rep_fake)
            netP.train()

            x_prob_fake = F.softmax(x_out_fake, dim=1)
            x_maxprob_fake,x_label_fake = x_prob_fake.max(dim=1)
            select_indices = x_maxprob_fake>opt.softmaxthe
            ys_rep_fake = y_rep_fake[select_indices,:,:,:]
            if(ys_rep_fake.shape[0]==0):
                errP_x = 0
            else:
                ys_label = x_label_fake[select_indices].detach()
                ys_out_fake = netP.pred_fromrep(ys_rep_fake, target=ys_label)
                if ys_rep_fake.shape[0]==1:
                    ys_out_fake = ys_out_fake[None, :]
                errP_x = criterion(ys_out_fake,ys_label) * opt.netPlossscale * 0.1


            optloss = ((x_rep_fake-y_rep_fake)**2).sum()/opt.batchSize * opt.OPTlossscale
            errP_x += criterion(x_out,x_label) * opt.netPlossscale
            # GAN training for y
            lossP = errD_x_real+errD_y_real+errD_x_fake+errD_y_fake+optloss+errP_x+errD_xy
            lossP.backward()
            optimizerP.step()

        ###### Solver outter
        netG.zero_grad()
        if opt.resample:
            z_noise = torch.randn(opt.batchSize, opt.zsize).to(device)
            x_fake, y_fake = netG(z_noise)
        x_out_d_fake,_,x_rep_fake,y_out_d_fake,_,y_rep_fake = netP(x_fake, y_fake)
        errD_x_fake = ganloss(x_out_d_fake) * opt.netDlossscale
        errD_y_fake = ganloss(y_out_d_fake) * opt.netDlossscale
        D_x_fake = x_out_d_fake.mean().item()
        D_y_fake = y_out_d_fake.mean().item()

        # train optimal transport loss
        optloss = ((x_rep_fake-y_rep_fake)**2).sum()/opt.batchSize * opt.OPTlossscale #+ opt_contour_loss(x_fake,y_fake) * opt.OPTlossscale2

        # Total Loss
        total_loss = errD_x_fake+errD_y_fake+optloss
        total_loss.backward()

        optimizerG.step()

        pred = torch.argmax(x_out,dim=-1)
        train_acc = torch.sum(pred==x_label).item()/len(pred)

        if i%100==0:
            print('Iter: [%d/%d] D_x_real: %.4f, D_x_fake: %.4f, D_y_real: %.4f, D_y_fake: %.4f, Loss_GANx: %.4f, Loss_GANy: %.4f, Loss_OPT: %.4f, Loss_P: %.4f, Train Accu: %.4f' %
                (i, opt.niter, D_x_real, D_x_fake, D_y_real, D_y_fake, errD_x_real.item()+errD_x_fake.item(), errD_y_real.item()+errD_y_fake.item(), optloss.item(), errP_x.item(), train_acc))

        netP.eval()
        # show tsne
        if i%opt.testiter == 0:
            if fixed_x is None:
                fixed_x = x.clone()
                fixed_y = y.clone()
                fixed_xlabel = x_label.to("cpu").long().numpy()
                fixed_ylabel = y_label.to("cpu").long().numpy()
                if fixed_x.shape[1] == fixed_y.shape[1]:
                    real_images = torch.cat((fixed_x, fixed_y), 2)
                elif fixed_x.shape[1] == 1:
                    real_images = torch.cat((torch.cat((fixed_x,fixed_x,fixed_x), 1), fixed_y), 2)
                else:
                    real_images = torch.cat((fixed_x, torch.cat((fixed_y,fixed_y,fixed_y), 1)), 2)
                torchvision.utils.save_image(real_images.data, opt.experiment +'/realimage.jpg')
            _,fixedx_rep = netP.pred_s(fixed_x)
            _,fixedy_rep = netP.pred_t(fixed_y)
            fixed_x_fake, fixed_y_fake = netG(fixed_noise)
            fixedx_rep_fake_l,fixedx_rep_fake = netP.pred_s(fixed_x_fake)
            fixedx_rep_fake_l = fixedx_rep_fake_l.argmax(dim=1)
            fixedy_rep_fake_l,fixedy_rep_fake = netP.pred_t(fixed_y_fake)
            fixedy_rep_fake_l = fixedy_rep_fake_l.argmax(dim=1)
            if fixed_x_fake.shape[1] == fixed_y_fake.shape[1]:
                fake_images = torch.cat((fixed_x_fake, fixed_y_fake), 2)
            elif fixed_x_fake.shape[1] == 1:
                fake_images = torch.cat((torch.cat((fixed_x_fake,fixed_x_fake,fixed_x_fake), 1), fixed_y_fake), 2)
            else:
                fake_images = torch.cat((fixed_x_fake, torch.cat((fixed_y_fake,fixed_y_fake,fixed_y_fake), 1)), 2)
            torchvision.utils.save_image(fake_images.data, opt.experiment +'/fakeimage_%05d.jpg'%(i))
            show_tsne(
                fixedx_rep.to("cpu").detach().numpy(),
                fixed_xlabel,
                fixedy_rep.to("cpu").detach().numpy(),
                fixed_ylabel,
                fixedx_rep_fake.to("cpu").detach().numpy(),
                fixedx_rep_fake_l,
                fixedy_rep_fake.to("cpu").detach().numpy(),
                fixedy_rep_fake_l,
                i)

        #     do checkpointing
        if (i+1) % opt.testiter == 0:
            test_acc = test_f()
            if best_test_acc < test_acc:
                best_test_acc = test_acc
                torch.save(netP.state_dict(), '{0}/netP.pth'.format(opt.experiment, i))

if __name__ == '__main__':
    main()
