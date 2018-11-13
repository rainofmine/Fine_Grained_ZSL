# First version of ZSL in Tianchi
from dataloader import *
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import config as cfg

from models.nts_net import *
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
#model = resnet101(300).cuda()
#model = LDF_baseline(Attr_164).cuda()
#model = resnet_mod56(164).cuda()
#model = ResNet(300).cuda()
model = attention_net(att_num=300, topN=cfg.PROPOSAL_NUM).cuda()


#-------------data--------------------
word_mat = torch.Tensor(np.load('./pkl/word_mat.npy')).cuda()
att_mat = torch.Tensor(np.load('./pkl/att_mat.npy')).cuda()

train_data = Trainset(os.listdir(cfg.IMG_PATH))
train_loader = DataLoader(train_data, cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.Lr, weight_decay=cfg.WEIGHT_DECAY)
#optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.Lr, momentum=0.9, weight_decay=cfg.WEIGHT_DECAY)
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

        raw_logits, concat_logits, part_logits, _, top_n_prob = model(batch_x)

        b = part_logits.size(0)
        if cfg.SEMANTIC == 'word':
            raw_scores = torch.mm(raw_logits, word_mat)
            concat_scores = torch.mm(concat_logits, word_mat)

            n = word_mat.size(1)
            part_scores = torch.mm(part_logits.view(-1, 300), word_mat).view(b, -1, n)

        elif cfg.SEMANTIC == 'att':
            raw_scores = torch.mm(raw_logits, att_mat)
            concat_scores = torch.mm(concat_logits, att_mat)
            n = word_mat.size(1)
            part_scores = torch.mm(part_logits.view(-1, 50), att_mat).view(b, -1, n)
        else:
            raise ValueError

        part_loss = list_loss(part_scores.view(b * cfg.PROPOSAL_NUM, -1),
                                    batch_y.unsqueeze(1).repeat(1, cfg.PROPOSAL_NUM).view(-1)).view(b, cfg.PROPOSAL_NUM)
        raw_loss = criterion(raw_scores, batch_y)
        concat_loss = criterion(concat_scores, batch_y)
        rank_loss = ranking_loss(top_n_prob, part_loss).squeeze()
        partcls_loss = criterion(part_scores.view(b * cfg.PROPOSAL_NUM, -1),
                                 batch_y.unsqueeze(1).repeat(1, cfg.PROPOSAL_NUM).view(-1))

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        lr_run = adjust_learning_rate(optimizer, i_iter=iter)

        writer.add_scalar("Train/learning_rate", lr_run, iter)
        writer.add_scalar("Train/Loss", total_loss, iter)
        if iter % 20 == 0:
            print('epoch: {} | step: {} | loss: {:.4f} | lr: {}'.format(epoch, iter, total_loss, lr_run))

    if epoch != 0 and epoch % 10 == 0:
        #------------valid-----------
        acc = 0
        correct = 0.
        total = 0.
        for data in train_loader:
            batch_x, batch_y = data

            batch_y = batch_y.long()
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            model.eval()
            _, concat_logits, _, _, _ = model(batch_x)

            if cfg.SEMANTIC == 'word':
                res = torch.mm(concat_logits, word_mat)
            elif cfg.SEMANTIC == 'att':
                res = torch.mm(concat_logits, att_mat)
            else:
                raise ValueError

            _, pre = torch.max(res, 1)

            total += batch_y.size(0)
            correct += (pre == batch_y).sum()

        correct = correct.cpu().numpy()
        acc = correct / total
        print('train acc for epoch {} : {:.4f} '.format(epoch, acc))
        writer.add_scalar("train/acc", acc, iter)

        if cfg.VERSION not in os .listdir(cfg.CKPT_PATH):
            os.mkdir(os.path.join(cfg.CKPT_PATH, cfg.VERSION))
        torch.save(model, os.path.join(cfg.CKPT_PATH, cfg.VERSION, 'epoch{}_{:.4f}.pkl'.format(epoch, acc)))


