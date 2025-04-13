import os.path
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms, datasets
from tqdm import tqdm

from deffusionDemo import Unet


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.002

    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.002

    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.002

    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(inputs, t_list, x_shape):
    batch, *_ = t_list.shape
    out = inputs.gather(-1, t_list)
    return out.reshape(batch, *((1,) * (len(x_shape) - 1)))


@torch.no_grad()
def predict_sample(model, img, t_list, t_idx):
    betas_t = extract(betas, t_list, img.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t_list, img.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t_list, img.shape)

    model_mean = sqrt_recip_alphas_t * (
            img - betas_t * model(img, t_list) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_idx == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t_list, img.shape)
        noise = torch.randn_like(img)
        # Algorithm 2 line 4
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def predict_sample_loop2(batches, batch_size=128):
    data = batches.to(device)
    label = None
    forward = 30
    t_list = torch.full((batch_size,), forward, device=device, dtype=torch.long)
    img = get_sample(data, t_list, noise=torch.randn_like(data))
    img_list = []

    for i in range(forward - 1, -1, -1):
        t_list = torch.full((batch_size,), i, device=device, dtype=torch.long)

        img = predict_sample(model, img, t_list, 1)
        img_list.append(img.cpu().numpy())
    return {'imgs': img_list,
            'label': label,
            't_list': t_list,
            'batches': batches}


@torch.no_grad()
def predict_sample_loop(model, image_size, batch_size=16, channels=3):
    device = next(model.parameters()).device
    img = torch.randn((batch_size, channels, image_size, image_size), device=device)
    imgs = []

    for i in tqdm(reversed(range(0, time_steps)), desc='sample loop time step', total=time_steps):
        img = predict_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())

    return {'imgs': imgs}


#  获取正向传播的各样本， 同一个样本不同的传播次数中使用的原噪声相同，只是系数不同
def get_sample(data, t_list, noise=None):
    if noise is None:
        noise = torch.randn_like(data)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t_list, data.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t_list, data.shape
    )

    return sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise


def calculate_loss(denoise_model, data, noise=None, loss_type='l1'):
    data = data.unsqueeze(1)
    batch_size = data.shape[0]
    if noise is None:  # true
        noise = torch.randn_like(data)  # [128, 1, 28, 28]

    t_list = torch.randint(0, time_steps, (batch_size,), device=device).long()
    sample_list = get_sample(data=data, t_list=t_list, noise=noise)  #
    predicted_noise = denoise_model(sample_list, t_list)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)  # 这里的noise好像没有做到随着t变而变，只有predicted做到了随着t变而变
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


def save_img(data_list, labels_list, batches):
    for data in data_list[-10:]:

        fig, axs = plt.subplots(1, 13, figsize=(33, 3))
        for i, (ax, data_img) in enumerate(zip(axs.flat, data[::10]), 1):
            data_img = np.squeeze(data_img)
            ax.axis('off')
            ax.imshow(data_img)
        plt.show()
        plt.close()

    fig, axs = plt.subplots(1, 13, figsize=(33, 3))
    data = batches[0].cpu().numpy()
    for idx, (ax, data_img) in enumerate(zip(axs.flat, data[::10]), 1):
        data_img = np.squeeze(data_img)
        ax.axis('off')
        ax.imshow(data_img)
    plt.show()
    plt.close()


if __name__ == '__main__':
    time_steps = 600  # 对于步骤， 一开始可以由beta、分布的均值和标准差来共同确定
    image_size = 28
    channels = 1
    batch_size = 128
    device = torch.device('cuda:0')
    # 获取每一步的beta
    betas = linear_beta_schedule(timesteps=time_steps).to(device)
    dataset = datasets.MNIST(root='../../dataset/mnist/',
                             train=True,
                             download=False,
                             transform=transforms.ToTensor())
    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)

    # 按照标签(label)将数据进行分组
    grouped_data = defaultdict(list)
    for dataset, label in data_loader:
        for idx, data in zip(label, dataset):
            grouped_data[idx.item()].append(data)

    data_loaders = []
    for label in range(10):
        subset = Subset(torch.cat(grouped_data[label], dim=0), range(len(grouped_data[label])))
        data_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        data_loaders.append(data_loader)

    # 计算alpha、 alpha_prod、 alpha_prod_previous、alpha_bar_sqrt 等变量的值
    # data_loader = data_loaders[0]
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 返回依次向后累成结果
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)  # 在idx=0处塞进一个1，并去掉最后一位
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)  # 根号下a分之一
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # 根号下a_bar
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # σ^2
    epochs = 600

    for label, data_loader in enumerate(data_loaders, 0):
        model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4,)
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # for epoch in tqdm(reversed(range(epochs)), total=epochs, desc="training"):
        for epoch in range(epochs):
            for step, batch in enumerate(data_loader):
                optimizer.zero_grad()
                batches = batch.to(device)
                loss = calculate_loss(model, batches, loss_type='huber')
                if step % 100 == 0:
                    print('\nloss:', loss.item())
                loss.backward()
                optimizer.step()
        # torch.save(batch_model, f'./denoise_diffusion_model{time_steps}+{idx}.pth')
        use_forward_data = False
        if use_forward_data:
            batches = next(iter(data_loader))
            batches = batches.unsqueeze(1)
            samples = predict_sample_loop2(batch_size=128, batches=batches)
            data = samples['imgs']
            labels = samples['label']
            save_img(data, labels, batches=batches)
        else:
            samples = predict_sample_loop(model, image_size, batch_size, channels=channels)
            data = samples['imgs'][-1]
            data = [np.squeeze(i) for i in data]
            fig, axs = plt.subplots(16, 8, figsize=(24, 48))  # 16行8列
            for idx in range(16 * 8):
                axs[idx // 8, idx % 8].axis('off')
                axs[idx // 8, idx % 8].imshow(data[idx])
            current_directory = os.path.dirname(os.path.abspath(__file__))
            sav_dir = os.path.join(current_directory, 'result')
            if not os.path.exists(sav_dir):
                os.mkdir(sav_dir)
            fig_path = f'{sav_dir}/time_steps{time_steps}epoch{epochs}+{label}.png'
            plt.savefig(fig_path)
            plt.show()
            plt.close()

