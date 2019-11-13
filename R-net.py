import Nets
import train_nets

if __name__ == '__main__':
    net = Nets.RNet()
    trainer = train_nets.Trainer(net, './param/rnet1.pth', r'E:\data\data\24')
    trainer.train()
