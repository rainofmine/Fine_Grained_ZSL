import os
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import config as cfg
from dataloader import img_normalizer
from PIL import Image
from tqdm import *

model_name = 'epoch60_0.9920.pkl'
model_path = os.path.join(cfg.CKPT_PATH, cfg.VERSION, model_name)
model = torch.load(model_path).cuda()
model.eval()

f = open('./submit/submit1021_pretrain.txt', 'w')

ftest = open('../DatasetMerged/image.txt')
test_list = [i[:-1] for i in ftest.readlines()]
word_mat_submit = torch.Tensor(np.load('./pkl/word_mat_submit.npy')).cuda()
label_ZJL_submit = pickle.load(open('./pkl/label_ZJL_submit.pkl', 'rb'))

for i in tqdm(range(len(test_list))):
    ads = os.path.join(cfg.TEST_PATH, test_list[i])
    img = Image.open(ads)
    tensor = img_normalizer(img)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.cuda()

    out = model(tensor)

    res = torch.mm(out, word_mat_submit)

    res_soft = F.softmax(res, 1)
    v, pre = torch.max(res_soft, 1)

    label = pre.cpu().numpy()[0]

    cls = label_ZJL_submit[label]
    f.write(str(test_list[i]) + '\t' + str(cls) + '\n')

f.close()



