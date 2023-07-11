# -*- coding:utf-8 -*-
import os
import time
import torch
import shutil
import nibabel as nib
from custom_losses import dice_coefficient
from custom_losses import dice_coefficient2


#https://github.com/pytorch/examples/blob/b9f3b2ebb9464959bdbf0c3ac77124a704954828/imagenet/main.py#L359
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


#https://github.com/pytorch/examples/blob/b9f3b2ebb9464959bdbf0c3ac77124a704954828/imagenet/main.py#L359
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_images(input, target, pred, affine, count, batch_size, save_path,names):
    for i in range(batch_size):
        current_affine = affine[i, :, :]
        current_affine = current_affine.cpu().detach().numpy()
        current_input = input[i, 0, :, :, :]
        current_input = current_input.permute(2, 1, 0)
        current_input = torch.squeeze(current_input)
        current_input = current_input.cpu().detach().numpy()
        current_input = nib.Nifti1Image(current_input, affine=current_affine)
        #nib.save(current_input, os.path.join(save_path, 'InputTestNorm_' + str(count) + str(i) + '.nii.gz'))
        nib.save(current_input, os.path.join(save_path, 'InputTestNorm_' + str(count)+'_' + names[i] ))
        # Assume that 0 is the background, 1 liver, 2 tumour
        current_target = torch.argmax(target[i, :, :, :, :], dim=0).float()
        current_target = current_target.permute(2, 1, 0)
        current_target = torch.squeeze(current_target)
        current_target = current_target.cpu().detach().numpy()
        current_target = nib.Nifti1Image(current_target, affine=current_affine)
        #nib.save(current_target, os.path.join(save_path, 'TargetTest_' + str(count) + str(i) + '.nii.gz'))
        nib.save(current_target, os.path.join(save_path, 'TargetTest_' + str(count)+'_' + names[i]))
        # Assume that 0 is the background, 1 liver, 2 tumour
        current_pred = torch.argmax(pred[i, :, :, :, :], dim=0).float()
        current_pred = current_pred.permute(2, 1, 0)
        current_pred = torch.squeeze(current_pred)
        current_pred = current_pred.cpu().detach().numpy()
        current_pred = nib.Nifti1Image(current_pred, affine=current_affine)
        #nib.save(current_pred, os.path.join(save_path, 'PredTest_' + str(count) + str(i) + '.nii.gz'))
        nib.save(current_pred, os.path.join(save_path, 'PredTest_' + str(count)+'_' + names[i] ))


#def adjust_learning_rate(optimizer, epoch, learning_rate, lr_decay_fact, lr_decay_time):
#    """Decrease the Learning Rate by multiplier on given number of epoch"""
#    lr = learning_rate * (lr_decay_fact ** (epoch // lr_decay_time))
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, device, verbose=False, dgminfo=None):
    batch_time = AverageMeter('Batch', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    dices = AverageMeter('Dice', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, dices],
        prefix="Train / Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input, target, affine,name = data['image'], data['mask'], data['affine'],data['name']
        # Print data samples to tensorboard
        #data_time.update(time.time() - end)
        #images = input[0][0][slice2print][:][:]
        #masks = target[0][0][slice2print][:][:]
        #img_grid = torchvision.utils.make_grid(images, scale_each=True, normalize=True)
        #plt.imshow(images, cmap="Greys")
        #writer.add_image('input', img_grid)
        #img_grid = torchvision.utils.make_grid(masks, scale_each=True, normalize=True)
        #plt.imshow(masks, cmap="Greys")
        #writer.add_image('Target', img_grid)

        # send data to device
        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)  # [batch, labels, z, y, x]
        
        output.shape

        # measure dice
        dices.update(dice_coefficient2(output, target).item(), input.size(0))

        # update loss/opti
        if not dgminfo:
            loss = criterion(output, target)
        else:
            loss = criterion(output, target,dgminfo)
            
        losses.update(loss.item(), input.size(0)) ##########

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print output sample
        #preds = torch.round(output)

        # print(preds.size())
        # print(target.size())
        #o = preds.cpu().detach()
        #o = o[0][0][slice2print][:][:]
        #img_grid = torchvision.utils.make_grid(o, scale_each=True, normalize=True)
        #plt.imshow(o, cmap="Greys")
        #writer.add_image('Prediction', img_grid)

        # measure accuracy, dice and record loss

        # ------Dice and CE Losses-----------
        # target = torch.unsqueeze(target, 1)

        # ------------For CELoss Only----------------
        # a = torch.arange(1,3)
        # a = a.view(1,2,1,1,1)
        # a = a.to(device)
        # target = torch.mul(a,target)
        # ----------------------------------------
        # print(a.size())
        # print(target.size())

        #prec1 = preds.eq(target).sum().item() / target.nelement()

        #dice = dice_score(preds=preds, target=target)
        #dice = torch.mean(dice)
        #dice = dice.item()

        #losses.update(loss.item(), input.size(0))
        #acc.update(prec1, input.size(0))
        #dsc.update(dice, input.size(0))

        # measure elapsed time
        #batch_time.update(time.time() - end)
        #end = time.time()

        # print statistics
        if verbose:
            progress.display(i)

        # Print metrics
        #if i % 10 == 0:
        #    print('Epoch: [{0}][{1}/{2}]\t'
        #          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #          'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(
        #        epoch, i, len(train_loader), batch_time=batch_time,
        #        data_time=data_time, loss=losses, acc=acc))
        #    niter = epoch * len(train_loader) + i
        #    writer.add_scalar('Train/Loss', losses.val, niter)
        #    writer.add_scalar('Train/Accuracy', acc.val, niter)
        #    writer.add_scalar('Train/DSC', dsc.val, niter)
        #    # writer.add_graph(model, input)
    return losses.avg, dices.avg


def val_test(val_test_loader, model, criterion, device, save_path=None, verbose=False, dgminfo=None):
    batch_time = AverageMeter('Batch', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    dices = AverageMeter('Dice', ':.4e')
    progress = ProgressMeter(
        len(val_test_loader),
        [batch_time, data_time, losses, dices],
        prefix="Val / Test:")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_test_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            input, target, affine,name = data['image'], data['mask'], data['affine'],data['name']

            # send data to device
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)

            # Save the images
            if save_path:
                names=val_test_loader.dataset.Images_Names
                save_images(input, target, output, affine, i, input.size(0), save_path,name)

            # measure dice
            dices.update(dice_coefficient2(output, target).item(), input.size(0))

            # update loss
            if not dgminfo:
                loss = criterion(output, target)
            else:
                loss = criterion(output, target,dgminfo)
            
            #loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print statistics
            if verbose:
                progress.display(i)
    #
    return losses.avg, dices.avg
