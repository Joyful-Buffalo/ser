import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_s_curve
from torch.utils.data import DataLoader
from tqdm import tqdm


class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super().__init__()
        self.linear = nn.ModuleList([
            nn.Linear(2, num_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_units, num_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_units, num_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_units, 2),
        ])
        self.step_embeddings = nn.ModuleList([
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units),
        ])

    def forward(self, x_0, t):
        x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linear[2 * idx](x)
            x += t_embedding
            x = self.linear[2 * idx + 1](x)
        x = self.linear[-1](x)
        return x


def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_step):
    # 随机采样一个时刻t,为了提高训练效率，这里我确保t不重置
    # weights = torch.ones(n_steps).expand(batch_size, -1)
    # t = torch.multinomial(weights, num_samples=1, replacement=False)  # [batch_size, 1]

    # 对于一个batch_size 样本生成随机时刻t，覆盖到更多时刻的t
    batch_size = x_0.shape[0]
    t = torch.randint(0, n_step, size=(batch_size // 2,)).to(device)
    t = torch.cat([t, n_step - 1 - t], dim=0)
    t = t.unsqueeze(-1)

    # x0的系数
    a = alphas_bar_sqrt[t]

    # eps的系数
    aml = one_minus_alphas_bar_sqrt[t]

    # 生成随机噪音eps
    e = torch.randn_like(x_0).to(device)

    # 构造模型的输入
    x = x_0 * a + e * aml

    # 送入模型，得到t时刻的随机噪声预测值
    output = model(x, t.squeeze(-1).to(device))

    # 与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()


def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    # 从
    cur_x = torch.randn(shape).to(device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    t = torch.tensor([t]).to(device)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x).to(device)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample


# 计算任意时刻的x的采样值，基于x_0 和参数重整化技巧
def q_x(x_0, t):
    noise = torch.randn_like(x_0).to(device)  # noise是从正态分布中生成的随机噪声
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    # alphas_t = extract(alphas_bar_sqrt, t, x_0)  # 得到sqrt(alphas_bar[t]), x_0的作用是传入shape
    # alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)  # 得到sqrt(1 - alphas_bar[t])
    return alphas_t * x_0 + alphas_1_m_t * noise  # 在[0]的基础上添加噪声


if __name__ == '__main__':
    s_curve, _ = make_s_curve(10 ** 4, noise=0.1)
    s_curve = s_curve[:, [0, 2]] / 10.0
    device = torch.device('cuda')
    print('shape of moons:', np.shape(s_curve))
    data = s_curve.T
    fig, ax = plt.subplots()
    ax.scatter(*data, color='red', edgecolor='white')
    ax.axis('off')
    dataset = torch.Tensor(s_curve).float().to(device)
    fig.show()
    num_steps = 100  # 对于步骤， 一开始可以由beta、分布的均值和标准差来共同确定

    # 判断每一步的beta
    betas = torch.linspace(-6, 6, num_steps).to(device)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # 计算alpha、 alpha_prod、 alpha_prod_previous、alpha_bar_sqrt 等变量的值
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == alphas_bar_sqrt.shape == \
           one_minus_alphas_bar_log.shape == one_minus_alphas_bar_sqrt.shape
    print('all the same shape:', betas.shape)

    num_shows = 100
    fig, axs = plt.subplots(10, 10, figsize=(28, 28))
    plt.rc('text', color='blue')

    for _ in range(num_shows):
        j = _ // 10
        k = _ % 10
        q_i = q_x(dataset, torch.tensor([_ * num_steps // num_shows]))
        axs[j, k].scatter(q_i[:, 0].cpu(), q_i[:, 1].cpu(), color='red', edgecolor='white')
        axs[j, k].set_axis_off()
        axs[j, k].set_title('$ q(\mathbf{x}_{' + str(_ * num_steps // num_shows) + '})$')
    fig.show()

    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoch = 4000
    plt.rc('text', color='blue')

    model = MLPDiffusion(n_steps=num_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tqdm(total=num_epoch) as pbar:
        for t in range(num_epoch):
            loss = 0
            pbar.update(1)
            for idx, batch_x in enumerate(dataloader):
                loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt,
                                         one_minus_alphas_bar_sqrt, num_steps)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

            if t % 100 == 0:
                print(loss)
                x_seq = p_sample_loop(model, dataset.shape, num_steps,
                                      betas, one_minus_alphas_bar_sqrt)
                x_seq = [item.to('cpu') for item in x_seq]
                fig, axs = plt.subplots(1, 10, figsize=(28, 3))

                for i in range(1, 11):
                    cur_x = x_seq[i * 10].detach()
                    axs[i - 1].scatter(cur_x.cpu()[:, 0], cur_x[:, 1], color='red',
                                       edgecolor='white')
                    axs[i - 1].set_axis_off()
                    axs[i - 1].set_title('$q(\mathbf{x}_{' + str(i * 10) + '}) $')
                fig.show()
