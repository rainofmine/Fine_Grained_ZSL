import numpy as np
import pickle
import pandas as pd

normalize = False

txt_path = '../DatasetMerged/'
split_path = '../DatasetMerged/DataConfig/'

img_ZJL_dict = {}
img_ZJL_txt = open(txt_path + 'train.txt')
for line in img_ZJL_txt.readlines():
    line = line.strip().split()
    img_ZJL_dict[line[0]] = line[1]
f = open('./pkl/img_ZJL.pkl', 'wb')
pickle.dump(img_ZJL_dict, f)

ZJL_catogery_dict = {}
ZJL_catogery_txt = open(txt_path + 'label_list.txt')
for line in ZJL_catogery_txt.readlines():
    line = line.strip().split()
    ZJL_catogery_dict[line[0]] = line[1]
f = open('./pkl/ZJL_catogery.pkl', 'wb')
pickle.dump(ZJL_catogery_dict, f)

# word embedding
catogery_word_dict = {}
catogery_word_txt = open(txt_path + 'class_wordembeddings.txt')
for line in catogery_word_txt.readlines():
    line = line.strip().split()
    catogery_word_dict[line[0]] = [float(i) for i in line[1:]]
f = open('./pkl/catogery_word.pkl', 'wb')
pickle.dump(catogery_word_dict, f)

LabelEncInv = pd.read_csv(split_path + 'LabelEncInv.csv', index_col=0, names=['Index','Class'])['Class']
word_mat = []
for i in range(len(LabelEncInv)):
    ZJL = LabelEncInv[i]
    catogery = ZJL_catogery_dict[ZJL]
    word = catogery_word_dict[catogery]
    if normalize:
        mo = sum([i**2 for i in word]) ** 0.5
        word = [i/mo for i in word]
    word_mat.append(word)
np.save('./pkl/word_mat.npy', np.array(word_mat).transpose())

ZJL_label_test_dict = {}
word_mat_test = []
test_ZJL = np.load(split_path + 'arr_CatUnseen.npy')
for i, ZJL in enumerate(test_ZJL):
    ZJL_label_test_dict[ZJL] = i
    catogery = ZJL_catogery_dict[ZJL]
    word = catogery_word_dict[catogery]
    if normalize:
        mo = sum([i**2 for i in word]) ** 0.5
        word = [i/mo for i in word]
    word_mat_test.append(word)
np.save('./pkl/word_mat_test.npy', np.array(word_mat_test).transpose())
f = open('./pkl/ZJL_label_test.pkl', 'wb')
pickle.dump(ZJL_label_test_dict, f)

label_ZJL_submit_dict = {}
word_mat_submit = []
submit_ZJL = np.load(split_path + 'arr_CatUnAnnotd.npy')
for i, ZJL in enumerate(submit_ZJL):
    label_ZJL_submit_dict[i] = ZJL
    catogery = ZJL_catogery_dict[ZJL]
    word = catogery_word_dict[catogery]
    if normalize:
        mo = sum([i**2 for i in word]) ** 0.5
        word = [i/mo for i in word]
    word_mat_submit.append(word)
np.save('./pkl/word_mat_submit.npy', np.array(word_mat_submit).transpose())
f = open('./pkl/label_ZJL_submit.pkl', 'wb')
pickle.dump(label_ZJL_submit_dict, f)

# attribute
catogery_att_dict = {}
catogery_att_txt = open(txt_path + 'attributes_per_class.txt')
for line in catogery_att_txt.readlines():
    line = line.strip().split()
    catogery_att_dict[line[0]] = [float(i) for i in line[1:]]
f = open('./pkl/catogery_att.pkl', 'wb')
pickle.dump(catogery_att_dict, f)

LabelEncInv = pd.read_csv(split_path + 'LabelEncInv.csv', index_col=0, names=['Index','Class'])['Class']
att_mat = []
for i in range(len(LabelEncInv)):
    ZJL = LabelEncInv[i]
    att = catogery_att_dict[ZJL]
    att_mat.append(att)
np.save('./pkl/att_mat.npy', np.array(att_mat).transpose())

att_mat_test = []
test_ZJL = np.load(split_path + 'arr_CatUnseen.npy')
for i, ZJL in enumerate(test_ZJL):
    att = catogery_att_dict[ZJL]
    att_mat_test.append(att)
np.save('./pkl/att_mat_test.npy', np.array(att_mat_test).transpose())