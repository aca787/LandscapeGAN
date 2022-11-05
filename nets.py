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
      try:
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
                #Format batch
                right_real = data['right_images'].to(device)   
                right_embed = data['embedding'].to(device)
                wrong_embed = data['wrong_embedding'].to(device)
                #right_real = data[0].to(device)
                #right_embed=None
                
                b_size = right_real.size(0)
                real_labels = torch.full((b_size, ), real_label, dtype=torch.float, device=device)

                
                #########################################
                #### Real image, right text #############
                #Train with all-real batch
                self.netD.zero_grad()
                #Forward pass real batch through D
                output_d_real,_ = self.netD(right_real, right_embed)
                output_d_real = output_d_real.view(-1)
                errD_real = self.criterion(output_d_real, real_labels) 
                errD_real.backward()
                #########################################
                
                #########################################
                #### Right image, wrong text #############
                fake_labels = torch.full((b_size, ), fake_label, dtype=torch.float, device=device)
                output_d_wrong, _ = self.netD(right_real, wrong_embed)
                errD_wrong = self.criterion(output_d_wrong, fake_labels) * 0.5
                errD_wrong.backward()
                D_x=output_d_wrong.mean().item()
                #########################################
                
                #########################################
                ##### Fake image right text ####################
                ## Train with all-fake batch
                #Generate batch of latent vectors
                noise=torch.randn(b_size, nz, 1, 1, device=device)
                #Generate fake image batch with G
                fake = self.netG(noise, right_embed)
                #Classify all fake batch with D
                output_d_fake, _ = self.netD(fake.detach(), right_embed)
                output_d_fake = output_d_fake.view(-1)
                #Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output_d_fake,  fake_labels)   * 0.5            
                errD_fake.backward()
                #Update D
                self.optimizerD.step()
                #Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD = errD_real + errD_fake  +errD_wrong  
                ####################################################
                
                D_G_z1 = output_d_fake.mean().item()
                
                #############################
                # (2) Update G network: maximize log(D(G(z)))
                self.netG.zero_grad()
                output_d_fake, _ = self.netD(fake, right_embed) # Go through the discriminator with the fake image again 
                errG = self.criterion(output_d_fake, real_labels)
                errG.backward()
                self.optimizerG.step()
                
                ############
                #Output training stats
                if i%10==0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                        % (epoch, num_epochs, i, len(self.dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1))
                    # Save Losses for plotting later
                    G_losses.append(errG.item())
                    D_losses.append(errD.item())

                    
                    

                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (iters % 10 == 0) or ((epoch == num_epochs-1) and (i == len(self.dataloader)-1)):
                        with torch.no_grad():
                            check = torch.from_numpy(self.embedder.embed(["beach", "mountain", "forest","desert", "field"])).to(device)
                            
                            fake = self.netG(fixed_noise, check).detach().cpu()
                        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                    iters += 1
            if (epoch%5==0):
                self.save_checkpoint(self.netD, self.netG, './checkpoint', epoch)
      except Exception as e:
            print(e)
      finally:
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









#self.netG.zero_grad()
#fake = self.netG(noise, right_embed)
#
## Since we just updated D, perform another forward pass of all-fake batch through D
#output, activation_fake = self.netD(fake, right_embed)
#
#_, activation_real = self.netD(right_real, right_embed)
#activation_fake = torch.mean(activation_fake, 0)
#activation_real = torch.mean(activation_real, 0)
##Calculate G\s loss based on this output
#errG = self.criterion(output, real_labels) +\
#self.l2_coeff * self.l2_loss(activation_fake, activation_real.detach()) +\
#self.l1_coeff * self.l1_loss(fake, right_real)
##Calculate gradients for G
#errG.backward()
#D_G_z2=output.mean().item()
##Update G
#self.optimizerG.step()
