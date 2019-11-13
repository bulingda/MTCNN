from mydata import Datasets
import torch.nn as nn
import torch.optim as optim
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
torch.cuda.set_device(0)
writer = SummaryWriter()


class Trainer:
    def __init__(self, net, save_path, dataset_path, isCuda=True):
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda

        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()
        self.ldmk_loss_fn = nn.MSELoss()
        # self.writer = SummaryWriter(comment='loss')

        self.optimizer = optim.Adam(self.net.parameters())
        # 可以阶段性训练
        if os.path.exists(self.save_path):
            # print('模型存在，接着训练')
            self.net = torch.load(self.save_path)
            self.net.eval()

        if self.isCuda:
            self.net.cuda()

    def train(self):
        datasets1 = Datasets(self.dataset_path)
        dataloader = DataLoader(datasets1, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
        # num_workers是多线程读取数据,防止速度不匹配
        # drop_las设置为true是为了数据不能被512整除剩余的分给下一组
        episode = 0
        while True:
            self.net.train()
            for i, (img_data_loader, category_loader, offset_loader, ldmk_loader) in enumerate(dataloader):
                if self.isCuda:
                    img_data_loader = img_data_loader.cuda()
                    category_loader = category_loader.cuda()
                    offset_loader = offset_loader.cuda()
                    ldmk_loader = ldmk_loader.cuda()
                net_output_category, net_output_offset, net_output_ldmk = self.net(img_data_loader)

                output_category = net_output_category.view(-1, 1)
                """p,r,o网络输出的形状不一样所以要将图片变为NV结构，标签是NV，主要是改p网络的
                """
                output_offset = net_output_offset.view(-1, 4)
                output_ldmk = net_output_ldmk.view(-1, 10)

                # category_mask = torch.lt(category_loader, 2).view(-1).cpu().numpy()
                category_mask = torch.nonzero(category_loader < 2)[:, 0]
                # print(category_mask)
                # 用nonzero出来的是二位的
                # lt是取小于2的，part样本不参与分类损失计算
                category = category_loader[category_mask].view(-1, 1)
                # category = Variable(category, requires_grad=True)
                # 按照掩码取对应的数据
                output_category = output_category[category_mask]
                # 按照掩码取网络输出的数据
                # output_category = Variable(output_category)
                # try:
                #     cls_loss = self.cls_loss_fn(output_category, category)
                # except BaseException as e:
                #     print('发生异常：{}'.format(e))
                #     print(output_category.shape)
                #     print(category.shape)
                cls_loss = self.cls_loss_fn(output_category, category)
                # 计算网络分类损失

                offset_mask = torch.nonzero(category_loader > 0)[:, 0]
                # gt 是取置信度大于0的数，负样本不参与运算
                offset = offset_loader[offset_mask].view(-1, 4)
                # offset = Variable(offset, requires_grad=True)
                # 按照偏移量的掩码来取数据中的偏移量
                output_offset = output_offset[offset_mask]
                # output_offset = Variable(output_offset)
                offset_loss = self.offset_loss_fn(output_offset, offset)

                ldmk_mask = torch.nonzero(category_loader == 3)[:, 0]
                landmark = ldmk_loader[ldmk_mask].view(-1, 10)
                output_ldmk = output_ldmk[ldmk_mask]
                ldmk_loss = self.ldmk_loss_fn(output_ldmk, landmark)

                loss = cls_loss + offset_loss + ldmk_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.writer.add_scalar('Train', cls_loss, episode)
                # if (episode + 1) % 5 == 0:
                if i % 10 == 9:
                    print("episode:", episode, "batch:", i + 1, "loss:", loss.cpu().data.numpy(), "cs_loss:",
                          cls_loss.cpu().data.numpy(),
                          "offset_loss:", offset_loss.cpu().data.numpy(), "landmark_loss",
                          ldmk_loss.cpu().data.numpy())
                writer.add_scalar('loss', loss.item(), global_step=1)
            writer.close()
            # self.writer.add_graph()
            torch.save(self.net, self.save_path)
            print("这轮训练完了")
            episode += 1



























