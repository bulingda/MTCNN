import Nets
import train_nets

if __name__ == '__main__':
    net = Nets.ONet()
    trainer = train_nets.Trainer(net, './param/onet1.pth', r'E:\data\data\48')
    trainer.train()
