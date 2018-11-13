# First version of ZSL in Tianchi
from dataloader import *
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import config as cfg

from models.RACNN import RACNN
from models.imagenet_resnet import resnet101,resnet50
from model import ResNet,se_resnet,se_resnet_smaller
from models.LDF import LDF_baseline
from models.resnet_mod import resnet_mod56
import torch.nn.functional as F
from tensorboardX import SummaryWriter


cudnn.enabled = True
cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)


def pairwise_ranking_loss(preds, size_average = True):
    """
        preds:
            list of scalar Tensor.
            Each value represent the probablity of each class
                e.g) class = 3
                    preds = [logits1[class], logits2[class]]
    """
    if len(preds) <= 1:
        return 0
    else:
        losses = []
        for pred in preds:
            loss = []
            for i in range(len(pred)-1):
                rank_loss = (pred[i]-pred[i+1] + 0.1).clamp(min = 0)
                # 0.05 margin
                loss.append(rank_loss)
            loss = torch.sum(torch.stack(loss))
            losses.append(loss)
        losses = torch.stack(losses)
        if size_average:
            losses = torch.mean(losses)
        else:
            losses = torch.sum(losses)
        return losses


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**power)

def lr_warm_up(base_lr,iter,max_iter,power,up,keep):
    if iter < up:
        return base_lr * ((float(iter)/up) ** power)
    if iter >= up and iter < keep:
        return base_lr
    if iter >= keep:
        return base_lr * ((1-float(iter-keep)/(max_iter-keep))**power)

def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(cfg.Lr, i_iter, cfg.NUM_STEPS, cfg.POWER)

    #lr = lr_warm_up(Lr, i_iter, NUM_STEPS, POWER, UP, KEEP)
    for param_lr in optimizer.param_groups:
        param_lr['lr'] = lr
    return lr


#------model--------------
#model = resnet50(300).cuda()
#model = LDF_baseline(Attr_164).cuda()
#model = resnet_mod56(164).cuda()
#model = ResNet(300).cuda()
#model = se_resnet(50).cuda()
model = RACNN(num_classes=300).cuda()

# model_dict = model.state_dict()
# pretrained_dict = torch.load('./pretrained/resnet50-19c8e357.pth')
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k[:2] != 'fc'}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)


#-------------data--------------------
# word_mat = torch.Tensor(np.load('./pkl/word_mat.npy')).cuda()
# word_mat_test = torch.Tensor(np.load('./pkl/word_mat_test.npy')).cuda()
# att_mat = torch.Tensor(np.load('./pkl/att_mat.npy')).cuda()
# att_mat_test = torch.Tensor(np.load('./pkl/att_mat_test.npy')).cuda()
#
# train_data = Trainset(np.load(cfg.SPLIT_PATH + 'index_SeenTrn.npy'))
# train_loader = DataLoader(train_data, cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
#
# valid_data = Trainset(np.load(cfg.SPLIT_PATH + 'index_SeenVal.npy'))
# valid_loader = DataLoader(valid_data, cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
#
# test_data = Testset(np.load(cfg.SPLIT_PATH + 'index_UnseenTst.npy'))
# test_loader = DataLoader(test_data, cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


word_mat = torch.Tensor(np.load('./pkl/word_mat.npy')).cuda()
att_mat = torch.Tensor(np.load('./pkl/att_mat.npy')).cuda()

train_data = Trainset(os.listdir(cfg.IMG_PATH))
train_loader = DataLoader(train_data, cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.Lr, weight_decay=cfg.WEIGHT_DECAY)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.Lr, momentum=0.9, weight_decay=cfg.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss().cuda()

model.train()

if cfg.VERSION not in os.listdir(cfg.LOG_PATH):
    os.mkdir(os.path.join(cfg.LOG_PATH, cfg.VERSION))
writer = SummaryWriter(os.path.join(cfg.LOG_PATH, cfg.VERSION))

acc = 0
for epoch in range(2000):
    for i, data in enumerate(train_loader, 0):

        iter = i + epoch * len(train_loader)

        train_loss = 0.
        batch_x, batch_y = data
        batch_y = batch_y.long()

        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        model.train()

        out1, out2 = model(batch_x)

        if cfg.SEMANTIC == 'word':
            res1 = torch.mm(out1, word_mat)
            res2 = torch.mm(out2, word_mat)
        elif cfg.SEMANTIC == 'att':
            res1 = torch.mm(out1, att_mat)
            res2 = torch.mm(out2, att_mat)
        else:
            raise ValueError

        preds = []
        for i in range(batch_y.shape[0]):
            pred = [res1[i][batch_y[i]], res2[i][batch_y[i]]]
            preds.append(pred)
        rank_loss = pairwise_ranking_loss(preds)

        loss1 = criterion(res1, batch_y)
        loss2 = criterion(res2, batch_y)
        loss = loss1 + loss2 + rank_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_run = adjust_learning_rate(optimizer, i_iter=iter)

        writer.add_scalar("Train/learning_rate", lr_run, iter)
        writer.add_scalar("Train/Loss", loss, iter)
        if iter % 10 == 0:
            print('epoch: {} | step: {} | loss1: {:.4f} | loss2: {:.4f} | rank_loss: {:.4f} | lr: {}'.format(epoch, iter, loss1, loss2, rank_loss, lr_run))

    if epoch != 0 and epoch % 2 == 0:
        # ------------train-----------
        acc = 0
        correct = 0.
        total = 0.
        for data in train_loader:
            batch_x, batch_y = data

            batch_y = batch_y.long()
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            model.eval()
            out1, out2 = model(batch_x)

            if cfg.SEMANTIC == 'word':
                res1 = torch.mm(out1, word_mat)
                res2 = torch.mm(out2, word_mat)
            elif cfg.SEMANTIC == 'att':
                res1 = torch.mm(out1, att_mat)
                res2 = torch.mm(out2, att_mat)
            else:
                raise ValueError

            res = res2
            _, pre = torch.max(res, 1)

            total += batch_y.size(0)
            correct += (pre == batch_y).sum()

        correct = correct.cpu().numpy()
        acc = correct / total
        print('train acc for epoch {} : {:.4f} '.format(epoch, acc))
        writer.add_scalar("train/acc", acc, iter)

        # #------------valid-----------
        # acc = 0
        # correct = 0.
        # total = 0.
        # for data in valid_loader:
        #     batch_x, batch_y = data
        #
        #     batch_y = batch_y.long()
        #     batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        #     model.eval()
        #     out1, out2 = model(batch_x)
        #
        #     if cfg.SEMANTIC == 'word':
        #         res1 = torch.mm(out1, word_mat)
        #         res2 = torch.mm(out2, word_mat)
        #     elif cfg.SEMANTIC == 'att':
        #         res1 = torch.mm(out1, att_mat)
        #         res2 = torch.mm(out2, att_mat)
        #     else:
        #         raise ValueError
        #
        #     res = res2
        #     _, pre = torch.max(res, 1)
        #
        #     total += batch_y.size(0)
        #     correct += (pre == batch_y).sum()
        #
        # correct = correct.cpu().numpy()
        # acc = correct / total
        # print('valid acc for epoch {} : {:.4f} '.format(epoch, acc))
        # writer.add_scalar("valid/acc", acc, iter)
        #
        # #-----------test--------------
        # acc = 0
        # correct = 0.
        # total = 0.
        # for data in test_loader:
        #     batch_x, batch_y = data
        #
        #     batch_y = batch_y.long()
        #     batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        #     model.eval()
        #     out1, out2 = model(batch_x)
        #
        #     if cfg.SEMANTIC == 'word':
        #         res1 = torch.mm(out1, word_mat)
        #         res2 = torch.mm(out2, word_mat)
        #     elif cfg.SEMANTIC == 'att':
        #         res1 = torch.mm(out1, att_mat)
        #         res2 = torch.mm(out2, att_mat)
        #     else:
        #         raise ValueError
        #
        #     res = res2
        #     _, pre = torch.max(res, 1)
        #
        #     total += batch_y.size(0)
        #     correct += (pre == batch_y).sum()
        #
        # correct = correct.cpu().numpy()
        # acc = correct / total
        # print('test acc for epoch {} : {:.4f} '.format(epoch, acc))
        # writer.add_scalar("test/acc", acc, iter)

        if cfg.VERSION not in os .listdir(cfg.CKPT_PATH):
            os.mkdir(os.path.join(cfg.CKPT_PATH, cfg.VERSION))
        torch.save(model, os.path.join(cfg.CKPT_PATH, cfg.VERSION, 'epoch{}_{:.4f}.pkl'.format(epoch, acc)))


