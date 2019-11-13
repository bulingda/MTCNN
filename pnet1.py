import Nets_1
import train_nets

if __name__ == '__main__':
    net = Nets_1.PNet()
    trainer = train_nets.Trainer(net, './param/pnet11.pth', r'E:\data\data\12')
    trainer.train()