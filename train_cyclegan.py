import torch
import torchvision.transforms

from tqdm import tqdm
import os
from PIL import Image
from dataloader import get_data_loader
from models import Discriminator, CycleGenerator, AdversarialLoss, CycleConsistencyLoss

from options import CycleGanOptions


class Trainer:
    def __init__(self, opts):
        self.opts = opts

        #config dirs
        self.expdir = './cycle_gan'
        self.plotdir = os.path.join(self.expdir, 'plots')
        self.ckptdir = os.path.join(self.expdir, 'checkpoints')

        os.makedirs(self.plotdir, exist_ok = True)
        os.makedirs(self.ckptdir, exist_ok = True)

        #config data
        self.apple_trainloader, self.apple_testloader = get_data_loader('Apple', self.opts.batch_size, self.opts.num_workers)
        self.windows_trainloader, self.windows_testloader = get_data_loader('Windows', self.opts.batch_size, self.opts.num_workers)

        #config models

        ##apple->windows generator
        self.G_a2w = CycleGenerator(self.opts).to(self.opts.device)
        ##windows->apple generator
        self.G_w2a = CycleGenerator(self.opts).to(self.opts.device)

        generator_params = list(self.G_a2w.parameters()) + list(self.G_w2a.parameters())

        ##apple discriminator
        self.D_a = Discriminator(self.opts).to(self.opts.device)

        ##windows discriminator
        self.D_w = Discriminator(self.opts).to(self.opts.device)

        discriminator_params = list(self.D_a.parameters()) + list(self.D_w.parameters())

        #config optimizers
        self.G_optim = torch.optim.Adam(generator_params, lr=self.opts.lr, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(discriminator_params, lr = self.opts.lr, betas = (0.5, 0.999))

        #config training
        self.niters = self.opts.niters
        self.criterion = AdversarialLoss()
        self.cycle_consistency = CycleConsistencyLoss()

    def run(self):

        for i in range(self.niters):
            if i % self.opts.eval_freq == 0:
                self.eval_step(i)
            if i % self.opts.save_freq == 0:
                self.save_step(i)
            self.train_step(i)


    def train_step(self, epoch):
        self.G_w2a.train()
        self.G_a2w.train()

        self.D_a.train()
        self.D_w.train()

        apple_loader = iter(self.apple_trainloader)
        windows_loader = iter(self.windows_trainloader)

        #num_iters = min(len(self.apple_trainloader) // self.opts.batch_size, len(self.windows_trainloader) // self.opts.batch_size)
        num_iters = min(len(self.apple_trainloader), len(self.windows_trainloader))

        pbar = tqdm(range(num_iters))
        for i in pbar:
            self.D_optim.zero_grad()
            self.G_optim.zero_grad()

            #load data
            apple_data = next(apple_loader).to(self.opts.device)
            windows_data = next(windows_loader).to(self.opts.device)

            #####TODO:train discriminator on real data#####
            D_real_loss = 0.
            D_real_apple = self.D_a(apple_data)
            labels_real_apple = torch.ones((apple_data.shape[0], 1, 1, 1)).to(self.opts.device)
            loss_D_real_apple = self.criterion(D_real_apple, labels_real_apple)

            D_real_windows = self.D_w(windows_data)
            labels_real_windows = torch.ones((windows_data.shape[0], 1, 1, 1)).to(self.opts.device)
            loss_D_real_windows = self.criterion(D_real_windows, labels_real_windows)

            D_real_loss = loss_D_real_apple + loss_D_real_windows
            D_real_loss.backward()

            #####TODO:train discriminator on fake data#####
            D_fake_loss = 0.
            G_fake_apple = self.G_w2a(windows_data)
            D_fake_apple = self.D_a(G_fake_apple)
            labels_fake_apple = torch.zeros((windows_data.shape[0], 1, 1, 1)).to(self.opts.device)
            loss_D_fake_apple = self.criterion(D_fake_apple, labels_fake_apple)

            G_fake_windows = self.G_a2w(apple_data)
            D_fake_windows = self.D_w(G_fake_windows)
            labels_fake_windows = torch.zeros((apple_data.shape[0], 1, 1, 1)).to(self.opts.device)
            loss_D_fake_windows = self.criterion(D_fake_windows, labels_fake_windows)

            D_fake_loss = loss_D_fake_apple + loss_D_fake_windows
            D_fake_loss.backward()
            self.D_optim.step()

            #####TODO:train generator#####
            G_loss = 0.
            G_images_a2w = self.G_a2w(apple_data)
            D_a2w = self.D_a(G_images_a2w)
            loss_G_a2w = self.criterion(D_a2w, labels_real_apple)

            G_images_w2a = self.G_w2a(windows_data)
            D_w2a = self.D_a(G_images_w2a)
            loss_G_w2a = self.criterion(D_w2a, labels_real_windows)

            G_loss = loss_G_a2w + loss_G_w2a

            if self.opts.use_cycle_loss:
                forward_loss = self.cycle_consistency(self.G_w2a(self.G_a2w(apple_data)), apple_data)

                backward_loss = self.cycle_consistency(self.G_a2w(self.G_w2a(windows_data)), windows_data)

                G_loss += (forward_loss+backward_loss)
            G_loss.backward()
            self.G_optim.step()
            ##############################

            pbar.set_description('Epoch: {}, G_loss: {:.4f}, D_loss: {:.4f}'.format(epoch, G_loss.item(), D_real_loss.item() + D_fake_loss.item()))


    def eval_step(self, epoch):
        #####TODO: generate 16 images from apple to windows and windows to apple from test data and save them in self.plotdir#####
        self.G_a2w.eval()
        self.G_w2a.eval()

        test_apple = iter(self.apple_testloader)
        test_windows = iter(self.windows_testloader)

        for i in range(16):
            tensor_to_pil_img = torchvision.transforms.ToPILImage()

            img_apple = next(test_apple).to(self.opts.device)
            img_save_apple = tensor_to_pil_img(img_apple.squeeze())
            img_save_apple.save(os.path.join(self.plotdir, f'apple_e{epoch}_{i}.png'))

            translation_a2w = self.G_a2w(img_apple).reshape(3, 32, 32).permute(1, 2, 0).clamp(-1, 1).detach().cpu().numpy()
            translation_a2w = ((translation_a2w + 1)*127.5).astype('uint8')
            Image.fromarray(translation_a2w).save(os.path.join(self.plotdir, f'windowsGeneration_e{epoch}_{i}.png'))


            img_windows = next(test_windows).to(self.opts.device)
            img_save_windows = tensor_to_pil_img(img_windows.squeeze())
            img_save_windows.save(os.path.join(self.plotdir, f'windows_e{epoch}_{i}.png'))

            translation_w2a = self.G_a2w(img_windows).reshape(3, 32, 32).permute(1, 2, 0).clamp(-1, 1).detach().cpu().numpy()
            translation_w2a = ((translation_w2a + 1) * 127.5).astype('uint8')
            Image.fromarray(translation_w2a).save(os.path.join(self.plotdir, f'appleGeneration_e{epoch}_{i}.png'))

    def save_step(self, epoch):
        #####TODO: save models in self.ckptdir#####
        torch.save(self.G_a2w.state_dict(), f'{self.ckptdir}/G_a2w__{epoch}.pt')
        torch.save(self.D_a.state_dict(), f'{self.ckptdir}/D_a__{epoch}.pt')
        torch.save(self.G_w2a.state_dict(), f'{self.ckptdir}/G_w2a__{epoch}.pt')
        torch.save(self.D_w.state_dict(), f'{self.ckptdir}/D_w__{epoch}.pt')


if __name__ == '__main__':
    opts = CycleGanOptions()
    trainer = Trainer(opts)
    trainer.run()