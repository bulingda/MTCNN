import Nets
import train_nets

if __name__ == '__main__':
    net = Nets.PNet()
    trainer = train_nets.Trainer(net, './param/pnet1.pth', r'E:\data\data\12')
    trainer.train()


