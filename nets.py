import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.utils as vutils

#custom randomly initialized weights called on netG and netD
def weights_init(m):
    classname=m.__class__.__name__
    if classname.find("Conv")!=-1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu, nc, ngf, nz, embed_dim, projected_embed_dim):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.projected_embed_dim = projected_embed_dim
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz+self.projected_embed_dim, ngf * 8, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(ngf*8), 
            nn.ReLU(True), 
        #State size. (ngf*8)x4x4
        nn.ConvTranspose2d(ngf*8, ngf*4, 4,2,1, bias=False), 
        nn.BatchNorm2d(ngf*4), 
        nn.ReLU(True), 
        #state size. (ngf*4)x8x8, 
        nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False), 
        nn.BatchNorm2d(ngf*2),
        nn.ReLU(True), 
        #state size. (ngf*2)x16x16, 
        nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False), 
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        #state size. (ngf)x32x32, 
        nn.ConvTranspose2d(ngf,nc,4,2,1,bias=False), 
        nn.Tanh() )
        
        self.projection = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=self.projected_embed_dim),
            #nn.BatchNorm1d(num_features=self.projected_embed_dim),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self,z, embed):
        return self.main(self.encoder(embed, z))
        #return self.main(z)
    def encoder(self, embed_vector, z):
        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([projected_embed, z], 1)
        return latent_vector



class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf, embed_dim, projected_embed_dim=128):
        super(Discriminator, self).__init__()
        self.ngpu=ngpu
        self.projected_embed_dim = 128
        self.main = nn.Sequential(
            #input is (nc)x64x64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True), 
            #state size, ndfx32x32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ndf*2), 
            nn.LeakyReLU(0.2, inplace=True),
            #state size (ndf*2)x16x16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ndf*4), 
            nn.LeakyReLU(0.2, inplace=True),
            #state size (ndf*2)x16x16
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(ndf*8), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size, (ndf*8) x4 x4
            #nn.Conv2d(ndf*8 + self.projected_embed_dim,1, 4, 1, 0, bias=False), 
            #nn.Sigmoid()
        )
        self.projection = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=self.projected_embed_dim),
                                        #nn.BatchNorm1d(num_features=self.projected_embed_dim),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.main_end = nn.Sequential(
            # state size, (ndf*8) x4 x4
            nn.Conv2d(ndf*8 + self.projected_embed_dim,1, 4, 1, 0, bias=False), 
            nn.Sigmoid()
        )
    def forward(self, inp, embed):
        intermediate = self.main(inp)
        projected_embed = self.projection(embed)#.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        replicated_embed = projected_embed.repeat(4,4,1,1).permute(2,3,0,1)
        hidden_concat = torch.cat([intermediate, replicated_embed], 1)
        y = self.main_end(hidden_concat)
        return y.view(-1, 1).squeeze(1), intermediate
    def forward_without_embed(self, inp, embed):
        intermediate=[1]
        y = self.main(inp)
        return y.view(-1, 1).squeeze(1), intermediate
    
        

class GANTrainer:
    def __init__(self, netG, netD, dataloader, embedder, optimizerD, optimizerG, criterion):
        self.netG=netG
        self.netD=netD
        self.dataloader=dataloader
        self.optimizerD=optimizerD
        self.optimizerG=optimizerG
        self.criterion=criterion
        self.embedder=embedder
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.l1_coeff = 50
        self.l2_coeff = 100

    def save_checkpoint(self, netD, netG, dir_path, epoch):
        path = dir_path #os.path.join(dir_path, subdir_path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, epoch))

    def run(self, num_epochs, device, nz):
        real_label = 1
        fake_label = 0
        #Lists to keep track of progress
        img_list = []
        G_losses=[]
        D_losses=[]
        iters=0
        #Create batch of latent vectors that we will use to visualize the progression of the generator
        fixed_noise = torch.randn(5, nz, 1, 1, device=device)
        #For each epoch
        for epoch in range(num_epochs):
            for i, data in enumerate(self.dataloader, 0):
                ##################################
                # (1) Update D network: maximize log(D(x))+log(1-D(G(z)))
                ##################################
                #Train with all-real batch
                self.netD.zero_grad()
                #Format batch
                right_real = data['right_images'].to(device)   
                right_embed = data['embedding'].to(device)
                wrong_real = data['wrong_images'].to(device)
                #right_real = data[0].to(device)
                #right_embed=None
                
                b_size = right_real.size(0)
                label=torch.full((b_size, ), real_label, dtype=torch.float, device=device)

            

                #Forward pass real batch through D
                ########################################
                #### Real image, right text
                output,_ = self.netD(right_real, right_embed)
                #output = output.view(-1)
                #Calculate loss on all-real batch
                errD_real=self.criterion(output, label)
                errD_real.backward()
                #########################################
                #### Wrong image, right text
                outputs, _ = self.netD(wrong_real, right_embed)
                errD_wrong = self.criterion(outputs, label)
                errD_wrong.backward()
                wrong_score = outputs
                #########################################
                D_x=output.mean().item()

                #########################################
                #### Generate fake image 
                ## Train with all-fake batch
                #Generate batch of latent vectors
                noise=torch.randn(b_size, nz, 1, 1, device=device)
                fake = self.netG(noise, right_embed)
                #Generate fake image batch with G
                label.fill_(fake_label)

                ##### Fake image right text
                #Classify all fake batch with D
                output,_=self.netD(fake.detach(), right_embed)
                #Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)

                errD_fake.backward()
                
                

                errD = errD_fake+errD_real+errD_wrong
                #errD.backward()
                #Calculate the gradients for this batch, accumulated (summed) with previous gradients
                
                D_G_z1=output.mean().item()
                #Update D
                self.optimizerD.step()
                
                
                #############################
                # (2) Update G network: maximize log(D(G(z)))
                self.netG.zero_grad()
                fake = self.netG(noise, right_embed)
                label.fill_(real_label) #Fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output, activation_fake = self.netD(fake, right_embed)
                
                _, activation_real = self.netD(right_real, right_embed)
                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)
                #Calculate G\s loss based on this output
                errG = self.criterion(output, label) +\
                                self.l2_coeff * self.l2_loss(activation_fake, activation_real.detach()) +\
                                self.l1_coeff * self.l1_loss(fake, right_real)
                #Calculate gradients for G
                errG.backward()
                D_G_z2=output.mean().item()
                #Update G
                self.optimizerG.step()
                
                ############
                #Output training stats
                if i%10==0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, num_epochs, i, len(self.dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    # Save Losses for plotting later
                    G_losses.append(errG.item())
                    D_losses.append(errD.item())

                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (iters % 10 == 0) or ((epoch == num_epochs-1) and (i == len(self.dataloader)-1)):
                        with torch.no_grad():
                            check = torch.from_numpy(self.embedder.embed(["beach at night", "green mountain", "sunny forest","large desert", "yellow field"])).to(device)
                            
                            fake = self.netG(fixed_noise, check).detach().cpu()
                        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                    iters += 1
            if (epoch%5==0):
                self.save_checkpoint(self.netD, self.netG, './checkpoint', epoch)
        return self.netG, self.netD, G_losses, D_losses


def createNet(net_type, device,ngpu, num_channels, num_features,  embedding_dim, projected_embed_dirm, z_size=0, verbose=True, **kwargs):
    #Creating the Discriminator
    if net_type=="discriminator":
        net = Discriminator(ngpu,num_channels, num_features,embedding_dim, projected_embed_dirm).to(device)
    elif net_type=="generator":
        net = Generator(ngpu, num_channels, num_features, z_size, embedding_dim, projected_embed_dirm).to(device)
    else:
        raise ValueError("net_type must be either 'discriminator' or 'generator' ")

    #if (device.type == 'cuda')and (ngpu>1):
    #    net=nn.DataParallel(net, list(range(ngpu)))

    net.apply(weights_init)
    if(verbose):
        print(net)
    return net
