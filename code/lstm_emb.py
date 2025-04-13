import numpy as np
import torch
import torch.nn as nn
import numpy as py
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
kll_max = 0


def fill_missing_data(x):
    for i in range(1, 29):
        x[f'前{i}天客户流量'].fillna(0, inplace=True)

    y = x.fillna(0)
    return y


class MyDataSet(Dataset):
    def __init__(self, is_train=True, sr=0.7):
        self.data, self.target = self.read_data(is_train, sr)
        self.len = self.__len__()

    def __getitem__(self, idx):
        data = []
        t = []
        for i in range(idx, idx + USE_DAY):
            data.append(self.data[i])
        for i in range(idx, idx + OUTPUT_SIZE):
            t.append(self.target[i])
        return np.array(data), np.array(t)

    @staticmethod
    def read_data(is_train=True, sr=0.7):
        raw_data = pd.read_csv("Training.txt")
        raw_data = fill_missing_data(raw_data)
        scaler = MinMaxScaler()

        data = raw_data[
            ['客户流量'] + ['shop_id', 'location_id', 'per_pay', 'score', 'comment_cnt', 'shop_level', '星期']
            ].values
        data_normalized = scaler.fit_transform(data)
        target = scaler.fit_transform(raw_data['客户流量'].iloc[USE_DAY:].values.reshape(-1, 1))
        last = np.array([[0] * USE_DAY]).reshape(-1, 1)
        target = np.concatenate((target, last), axis=0)
        global kll_max
        kll_max = raw_data['客户流量'].max()
        return data_normalized[:int(len(raw_data) * sr)] if is_train else data_normalized[
                                                                          int(len(raw_data) * sr):], target

    def __len__(self):
        return len(self.data) - USE_DAY - OUTPUT_SIZE


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(64, 10)
        self.lstm = nn.LSTM(80, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size * USE_DAY, output_size)  # 将模型输出结果经过线性变化映射到输出层

    def forward(self, x):
        x = x.long()
        kll = self.embedding(x[:,:,0])
        shop_id = self.embedding(x[:, :, 1])
        ld = self.embedding(x[:, :, 2])
        pp = self.embedding(x[:, :, 3])
        score = self.embedding(x[:, :, 4])
        cc = self.embedding(x[:, :, 5])
        sl = self.embedding(x[:, :, 6])
        xq = self.embedding(x[:, :, 7])
        all_x = [kll, shop_id, ld, pp, score, cc, sl, xq]
        x = torch.cat(all_x, dim=-1)
        # x = torch.cat([x[:, :, 0:1], shop_id], dim=-1)
        # x = self.embedding(x)
        out, _ = self.lstm(x)

        out = out.reshape(-1, HIDDEN_SIZE * USE_DAY)
        out = self.fc(out)
        return out


def to_device(i, j):
    i = i.to(device).float().transpose(0, 1)
    j = j.to(device).float().transpose(0, 1).view(-1, 1)
    return i, j


def train():
    train_loss = []
    lstm_model.train()
    for inputs, t in train_loader:
        batches, t = to_device(inputs, t)
        output = lstm_model(batches)  # 前向传播
        loss = criterion(output, t)  # 计算loss
        optimizer.zero_grad()  # 把梯度清零
        loss.backward()  # loss反向传播
        optimizer.step()  # 更新模型参数
        train_loss.append(loss.item())
    return sum(train_loss) / len(train_loader)


def test():
    lstm_model.eval()
    test_loss = []
    with torch.no_grad():
        for inputs, t in test_loader:
            batches, t = to_device(inputs, t)
            output = lstm_model(batches)  # 前向传播
            loss = criterion(output, t)
            test_loss.append(loss.item())
    return sum(test_loss) / len(test_loader)


def my_plot():
    lstm_model.eval()
    with torch.no_grad():
        for inputs, t in train_loader:
            batches = inputs.to(device).float()
            t = t.to(device).float()
            output = lstm_model(batches)  # 前向传播
            output = [i.item() * kll_max for i in output]
            t = [i.item() * kll_max for i in t]
            plt.plot(range(len(output)), output)
            plt.plot(range(len(t)), t)
            plt.legend(['pred', 'target'])
            plt.show()
            break
    return output, t


def main(epochs):
    train_loss = []
    test_loss = []
    # pbar = tqdm(total=epochs, desc="Training")
    for idx in tqdm(range(epochs)):
        train_loss.append(train())
        test_loss.append(test())
        print(f"Epoch: {idx} Train_loss {train_loss[-1]}, Test_loss{test_loss[-1]}")
    mse = test()
    print("MSE", mse)
    o, t = my_plot()
    print(o, t)
    plt.plot(range(epochs), train_loss)
    plt.plot(range(epochs), test_loss)
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'test_loss'])
    plt.show()
    torch.save(lstm_model, f'{BATCH_SIZE}_lstm.pt')


if __name__ == '__main__':
    BATCH_SIZE = 64
    SHUFFLE = True
    EPOCHS = 200
    INPUT_SIZE = 8
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 1
    NUM_LAYERS = 1
    USE_DAY = 28
    USE_PRE = False
    train_data = MyDataSet(is_train=True)
    test_data = MyDataSet(is_train=False)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    if USE_PRE:
        lstm_model = torch.load(f'{BATCH_SIZE}_{EPOCHS}lstm.pt', map_location=device)
    else:
        lstm_model = Lstm(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    lstm_model = lstm_model.to(device)
    # lstm_model = nn.DataParallel(lstm_model)
    # loss
    criterion = nn.MSELoss()

    # 优化器,学习率为0.01
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=10 ** -3)
    main(EPOCHS)
