import torch
import torch.nn as nn
from torchvision import transforms, utils
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import numpy as np
import time, pprint, json, argparse, os, tqdm, re, random
import matplotlib.pyplot as plt
# from bisect import bisect_right
# import scipy.io as scio
# from PIL import Image
# from collections import defaultdict
from tensorboardX import SummaryWriter

from config import cfg
import network, dataloader, dataset
from utils import *

torch.backends.cudnn.benchmark = True


class MuDeep_v2():
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_cuda = self.cfg.GPU_ID is not None
        self.device = torch.device('cuda:{}'.format(self.cfg.GPU_ID[0])) if self.use_cuda else torch.device('cpu')
        self.cpu_device = torch.device('cpu')

        self.MODEL_PATH = os.path.join(self.cfg.ROOT, 'model')
        self.LOG_PATH = os.path.join(self.cfg.ROOT, 'log')
        self.set_random_seed(12345)
        self.build_model()   # build model
        self.root = self.cfg.ROOT
        self.name = self.cfg.NAME

        print('------------------------ Options -------------------------')
        for k, v in sorted(cfg.items()):
            if not isinstance(v, dict):
                print('%s: %s' % (k, v))
            else:
                print('%s: ' % k)
                for kk, vv in sorted(v.items()):
                    print('    %s: %s' % (kk, vv))
        print('-------------------------- End ----------------------------')

    def train(self, ):
        # save cfg to the disk during training
        self.check_file_exist(self.LOG_PATH)
        self.check_file_exist(self.MODEL_PATH)
        self.check_file_exist(os.path.join(self.MODEL_PATH, self.cfg.NAME))

        file_name = os.path.join(self.MODEL_PATH, self.cfg.NAME, 'opt.txt')
        self.opt_file = open(file_name, 'w')
        self.opt_file.write('------------------------ Options -------------------------\n')
        for k, v in sorted(cfg.items()):
            if not isinstance(v, dict):
                self.opt_file.write('%s: %s \n' % (k, v))
            else:
                self.opt_file.write('%s: \n' % k)
                for kk, vv in sorted(v.items()):
                    self.opt_file.write('    %s: %s \n' % (kk, vv))
        self.opt_file.write('-------------------------- End ----------------------------\n')
        self.opt_file.write('\n------------------------ Accuracy -------------------------\n')

        self.build_optimizer()

        train_data = dataloader.Market_Triplet_Loader(root=self.cfg.TRAIN.ROOT,
                                                       size=self.cfg.TRAIN.SIZE,
                                                       k=self.cfg.TRAIN.K,
                                                       repeat=3,
                                                       type='train')
        train_loader = Data.DataLoader(train_data, batch_size=self.cfg.TRAIN.K*self.cfg.TRAIN.P, shuffle=False, num_workers=8, drop_last=True)
        gallery_data = dataloader.Market_Loader(root=self.cfg.TEST.GALLERY,
                                             size=self.cfg.TRAIN.SIZE,
                                             type='gallery')
        gallery_loader = Data.DataLoader(gallery_data, batch_size=self.cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False)
        query_data = dataloader.Market_Loader(root=self.cfg.TEST.QUERY,
                                           size=self.cfg.TRAIN.SIZE,
                                           type='query')
        query_loader = Data.DataLoader(query_data, batch_size=self.cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False)

        for epoch in range(1, cfg.TRAIN.MAX_EPOCH + 1):
            epoch_loss = 0
            self.scheduler.step()
            train_data.preprocess()

            for step, data in enumerate(train_loader):
                begin = time.time()

                # #############################
                # (1) Data process
                # #############################
                img, label, path = data
                img = img.to(self.device)   # k*p x 3 x h x w
                label = label.to(self.device)   # k*p x 1

                # #############################
                # (2) Forward
                # #############################
                out_1_g_0, out_1_l_0, out_1_l_1, out_1_l_2, \
                out_2_g_0, out_2_l_0, out_2_l_1, out_2_l_2, \
                out_3_g_0, out_3_l_0, out_3_l_1, out_3_l_2, \
                feature1, feature2, feature3 = self.model(img)

                # #############################
                # (3) Loss
                # #############################
                positive1, negative1 = TripHard(feature1, p=self.cfg.TRAIN.P, k=self.cfg.TRAIN.K, norm=True)
                positive2, negative2 = TripHard(feature2, p=self.cfg.TRAIN.P, k=self.cfg.TRAIN.K, norm=True)
                positive3, negative3 = TripHard(feature3, p=self.cfg.TRAIN.P, k=self.cfg.TRAIN.K, norm=True)

                loss1 = self.loss_func(out_1_g_0, label) + self.loss_func(out_1_l_0, label) + \
                        self.loss_func(out_1_l_1, label) + self.loss_func(out_1_l_2, label)
                loss2 = self.loss_func(out_2_g_0, label) + self.loss_func(out_2_l_0, label) + \
                        self.loss_func(out_2_l_1, label) + self.loss_func(out_2_l_2, label)
                loss3 = self.loss_func(out_3_g_0, label) + self.loss_func(out_3_l_0, label) + \
                        self.loss_func(out_3_l_1, label) + self.loss_func(out_3_l_2, label)
                loss_CEL = loss1 / 4. + loss2 / 4. + loss3 / 4.              # Cross Entropy Loss
                loss_Tri1 = torch.mean(torch.clamp((positive1 - negative1 + cfg.TRAIN.MARGIN), min=0.0))  # TripHard Loss
                loss_Tri2 = torch.mean(torch.clamp((positive2 - negative2 + cfg.TRAIN.MARGIN), min=0.0))  # TripHard Loss
                loss_Tri3 = torch.mean(torch.clamp((positive3 - negative3 + cfg.TRAIN.MARGIN), min=0.0))  # TripHard Loss
                loss_Tri = loss_Tri1 + loss_Tri2 + loss_Tri3
                loss = 2 * loss_CEL + 1 * loss_Tri

                # #############################
                # (4) Display and Backward
                # #############################
                epoch_loss += loss.item()
                print ('Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  |  loss_CEL: {:.6f}  |  loss_Tri: {:.6f}  |  Time: {:.3f}'
                       .format(epoch, self.cfg.TRAIN.MAX_EPOCH, step+1, len(train_loader), self.optimizer.param_groups[0]['lr'],
                               loss_CEL.item(), loss_Tri.item(), time.time()-begin))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % self.cfg.TRAIN.SNAPSHOT == 0:
                # #############################
                # (5) Validate
                # #############################
                self.model.eval()
                with torch.no_grad():
                    gallery_info = self.extract_feature(gallery_loader, 'gallery')
                    query_info = self.extract_feature(query_loader, 'query')
                    cmc, map = self.evaluate(gallery_info, query_info)

                    print('-- Epoch:%d, Rank@1: %f, Rank@5: %f, Rank@10: %f, mAP: %f'%(epoch, cmc[0], cmc[4], cmc[9], map))
                    self.opt_file.write('Epoch:%d, Rank@1:%.4f, Rank@5:%.4f, Rank@10:%.4f, Rank@15:%.4f, Rank@20:%.4f, mAP:%.4f \n'
                                        %(epoch, cmc[0], cmc[4], cmc[9], cmc[14], cmc[19], map))
                    self.opt_file.flush()

                    self.summary.add_scalar('epoch_rank1', cmc[0], epoch)
                    self.summary.add_scalar('epoch_map', map, epoch)
                    self.summary.add_scalar('epoch_loss', epoch_loss/len(train_data), epoch)
                self.model.train()

            # #############################
            # (6) Save
            # #############################
            if epoch % self.cfg.TRAIN.CHECKPOINT == 0:
                self.save_model(epoch, self.MODEL_PATH, self.model, self.name, self.optimizer)

        self.summary.close()
        self.opt_file.close()

    def test(self,  model_path, out_name, overwrite=False):
        self.model.eval()
        is_folder = not os.path.splitext(model_path)[1] == '.pkl'

        gallery_data = dataloader.Market_Loader(root=self.cfg.TEST.GALLERY,
                                             size=self.cfg.TRAIN.SIZE,
                                             type='gallery')
        gallery_loader = Data.DataLoader(gallery_data, batch_size=self.cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False)
        query_data = dataloader.Market_Loader(root=self.cfg.TEST.QUERY,
                                           size=self.cfg.TRAIN.SIZE,
                                           type='query')
        query_loader = Data.DataLoader(query_data, batch_size=self.cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False)

        if is_folder:
            file = os.path.join(model_path, out_name + '.txt')
            if not overwrite:
                assert not os.path.exists(file), 'The output file alreay exists.'
            out_file = open(file, 'w')
            paths = [os.path.join(model_path, p) for p in os.listdir(model_path) if os.path.splitext(p)[1] == '.pkl']
            # paths.sort(key=lambda x:int(os.path.splitext(x)[0].split('_')[-1]))
            paths.sort(key=lambda x:int(os.path.splitext(x)[0].split('-')[0].split('_')[-1]))
        else:
            file = os.path.join(model_path.rsplit('/', 1)[0], out_name + '.txt')
            if not overwrite:
                assert not os.path.exists(file), 'The output file alreay exists.'
            out_file = open(file, 'w')
            paths = [model_path]

        out_file.write('------------------------ Accuracy -------------------------\n')

        with torch.no_grad():
            for path in paths:
                #  epoch = int(os.path.splitext(path)[0].split('_')[-1])
                epoch = int(os.path.splitext(path)[0].split('-')[0].split('_')[-1])
                self.model.load_state_dict(torch.load(path, map_location=self.device)['state_dict'])

                gallery_info = self.extract_feature(gallery_loader, 'gallery')
                query_info = self.extract_feature(query_loader, 'query')
                cmc, map = self.evaluate(gallery_info, query_info)

                print('-- Epoch:%d, Rank@1: %f, Rank@5: %f, Rank@10: %f, mAP: %f'%(epoch, cmc[0], cmc[4], cmc[9], map))
                out_file.write('Epoch:%d, Rank@1:%f, Rank@5:%f, Rank@10:%f, Rank@15:%f, Rank@20:%f, mAP:%f \n'
                                %(epoch, cmc[0], cmc[4], cmc[9], cmc[14], cmc[19], map))
                out_file.flush()
        out_file.close()

    def extract_feature(self, dataloader, type):
        features = torch.FloatTensor()
        cameras = []
        labels = []
        names = []
        for data in tqdm.tqdm(dataloader, desc='-- Extract %s features: ' % (type)):
            img, _, path = data
            label = [self.get_name(p) for p in path]
            camera = [self.get_camera(p) for p in path]
            name = [p for p in path]
            labels += label
            cameras += camera
            names += name

            n, c, h, w = img.size()
            ff = torch.FloatTensor(n, 512*12).zero_()

            for i in range(2):
                if (i==1):
                    img = self.fliplr(img)
                input_img = img.to(self.device)
                out_1_g_0, out_1_l_0, out_1_l_1, out_1_l_2, \
                out_2_g_0, out_2_l_0, out_2_l_1, out_2_l_2, \
                out_3_g_0, out_3_l_0, out_3_l_1, out_3_l_2 = self.model(input_img, test=True)
                outputs = torch.cat((out_1_g_0, out_1_l_0, out_1_l_1, out_1_l_2,
                                     out_2_g_0, out_2_l_0, out_2_l_1, out_2_l_2,
                                     out_3_g_0, out_3_l_0, out_3_l_1, out_3_l_2), dim=1)

                f = outputs.data.cpu()
                ff = ff + f / 2.
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff), 0)

        return {'feature': features,
                'camera': cameras,
                'label': labels,
                'name': names}

    def evaluate(self, gallery, query):
        query_feature = query['feature']
        query_cam = np.array(query['camera'])
        query_label = np.array(query['label'])
        gallery_feature = gallery['feature']
        gallery_cam = np.array(gallery['camera'])
        gallery_label = np.array(gallery['label'])

        query_feature = query_feature.to(self.device)
        gallery_feature = gallery_feature.to(self.device)

        # print(query_feature.shape)
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = self._evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            #print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC/len(query_label) #average CMC
        # print(len(CMC))
        # print('-- Rank@1: %f, Rank@5: %f, Rank@10: %f, mAP: %f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
        return CMC, ap/len(query_label)

    def _evaluate(self,qf,ql,qc,gf,gl,gc):
        query = qf.view(-1,1)
        # print(query.shape)
        score = torch.mm(gf,query)
        score = score.squeeze(1).to(self.cpu_device)
        score = score.numpy()
        # predict index
        index = np.argsort(score)  #from small to large
        index = index[::-1]
        # index = index[0:2000]
        # good index
        query_index = np.argwhere(gl==ql)
        camera_index = np.argwhere(gc==qc)

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)

        junk_index1 = np.argwhere(gl==-1)
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1) #.flatten())

        CMC_tmp = self.compute_mAP(index, good_index, junk_index)
        return CMC_tmp

    def build_model(self, ):
        self.model = network.MuDeep_v2(num_class=cfg.NUM_CLASS, num_scale=3, pretrain=True)
        self.model.train()
        if self.use_cuda:
            # self.model.to(self.device)
            self.model.to(self.cfg.GPU_ID[0])
            self.model = torch.nn.DataParallel(self.model, self.cfg.GPU_ID)
        print (self.model)

    def build_optimizer(self, ):
        ignored_params = list(map(id, self.model.module.scale1.parameters())) + list(map(id, self.model.module.scale1_1.parameters())) + list(map(id, self.model.module.scale1_2.parameters())) + list(map(id, self.model.module.scale1_3.parameters())) + \
                         list(map(id, self.model.module.scale2.parameters())) + list(map(id, self.model.module.scale2_1.parameters())) + list(map(id, self.model.module.scale2_2.parameters())) + list(map(id, self.model.module.scale2_3.parameters())) + \
                         list(map(id, self.model.module.scale3.parameters())) + list(map(id, self.model.module.scale3_1.parameters())) + list(map(id, self.model.module.scale3_2.parameters())) + list(map(id, self.model.module.scale3_3.parameters())) + \
                         list(map(id, self.model.module.atten1.parameters())) + list(map(id, self.model.module.atten2.parameters())) + list(map(id, self.model.module.atten3.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())
        self.optimizer = torch.optim.SGD([
                     {'params': base_params, 'lr': cfg.TRAIN.LR*0.1},
                     {'params': self.model.module.scale1.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.scale1_1.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.scale1_2.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.scale1_3.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.scale2.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.scale2_1.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.scale2_2.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.scale2_3.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.scale3.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.scale3_1.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.scale3_2.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.scale3_3.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.atten1.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.atten2.parameters(), 'lr': cfg.TRAIN.LR},
                     {'params': self.model.module.atten3.parameters(), 'lr': cfg.TRAIN.LR},
                ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        # self.optimizer = torch.optim.SGD(params=self.model.module.parameters(), lr=cfg.TRAIN.LR, weight_decay=5e-4, momentum=0.9, nesterov=True)
        self.scheduler = StepLR(self.optimizer, step_size=self.cfg.TRAIN.STEPSIZE, gamma=cfg.TRAIN.GAMMA)
        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.summary = SummaryWriter(log_dir='%s/%s' % (self.LOG_PATH, self.name), comment='')

    def check_file_exist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_model(self, epoch, path, model, name, optimizer):
        print ("-- Saving %d-th epoch model .......... " % (epoch), end='')
        if not os.path.exists(os.path.join(path, name)):
            os.mkdir(os.path.join(path, name))
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(), },
                    f='%s/%s/%s_%d.pkl' % (path, name, name, epoch))
        print ("Finished!")

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    @staticmethod
    def fliplr(img):
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long() # N x c x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    @staticmethod
    def get_name(image):
        return int(image.split('_')[0])

    @staticmethod
    def get_camera(image):
        return int(image.split('_')[1][1])

    @staticmethod
    def compute_mAP(index, good_index, junk_index):
        ap = 0
        cmc = torch.IntTensor(len(index)).zero_()
        if good_index.size==0:   # if empty
            cmc[0] = -1
            return ap,cmc

        # remove junk_index
        mask = np.in1d(index, junk_index, invert=True)
        index = index[mask]

        # find good_index index
        ngood = len(good_index)
        mask = np.in1d(index, good_index)
        rows_good = np.argwhere(mask==True)
        rows_good = rows_good.flatten()

        cmc[rows_good[0]:] = 1
        for i in range(ngood):
            d_recall = 1.0/ngood
            precision = (i+1)*1.0/(rows_good[i]+1)
            if rows_good[i]!=0:
                old_precision = i*1.0/rows_good[i]
            else:
                old_precision=1.0
            ap = ap + d_recall*(old_precision + precision)/2

        return ap, cmc


if __name__ == '__main__':
    engine = MuDeep_v2(cfg)

    engine.train()
    # engine.test(model_path='home/qxl/work/mudeep_v2/model/market',
    #             out_name='market_evaluate'
    #            )
