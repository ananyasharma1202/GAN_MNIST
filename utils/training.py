import torch
from torch.autograd import Variable
from utils.visualization import show_result, show_train_hist

def train(D, G, D_optimizer, G_optimizer, BCE_loss, train_loader, train_epoch, fixed_z_, log_f, config):
    train_hist = {'D_losses': [], 'G_losses': []}

    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []

        for x_, _ in train_loader:
            D.zero_grad()

            x_ = x_.view(-1, 28 * 28)
            mini_batch = x_.size()[0]

            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)

            x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)
            D_result = D(x_).reshape(-1)
            D_real_loss = BCE_loss(D_result, y_real_)
            D_real_score = D_result

            z_ = torch.randn((mini_batch, 100))
            z_ = Variable(z_)
            G_result = G(z_)

            D_result = D(G_result).reshape(-1)
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data)

            G.zero_grad()
            z_ = torch.randn((mini_batch, 100))
            y_ = torch.ones(mini_batch)

            z_, y_ = Variable(z_), Variable(y_)
            G_result = G(z_)
            D_result = D(G_result).reshape(-1)
            G_train_loss = BCE_loss(D_result, y_)
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data)

        epoch_log = '[%d/%d]: loss_d: %.3f, loss_g: %.3f\n' % (
            (epoch + 1), train_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses)))
        print(epoch_log)
        log_f.write(epoch_log)

        # Show result for every epoch
        show_result(G, epoch + 1, fixed_z_, show=False,  path=f"{config['training']['save_dir']}/Random_results/MNIST_GAN_{epoch + 1}.png", save=False, isFix=False)
        show_result(G, epoch + 1, fixed_z_, show=False, path=f"{config['training']['save_dir']}/Fixed_results/MNIST_GAN_{epoch + 1}.png", save=False, isFix=True)
       
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    return G, train_hist
