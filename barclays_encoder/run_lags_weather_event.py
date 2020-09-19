import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 'cuda'

import numpy as np
import pandas as pd

from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib import cm

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import torch.optim as optim
from torch.utils import data

# ---------------------------------------- GLOBAL PARAMETERS
NUM_PRED = 3
NUM_LAGS = 9
batch_size = 32
sel = [0, 2, 4, 5, 7, 8, 9]  # weather features to use
# sel = [5,7]
result_file = 'barclays_lags_weather_event_result.txt'

# word embeddings parameters

# ---------------------------------------- Load weather data
print("loading weather data...")

# load data
df = pd.read_csv("../central_park_weather.csv")
df = df.set_index("date")

# replace predefined values with NaN
df = df.replace(99.99, np.nan)
df = df.replace(999.9, np.nan)
df = df.replace(9999.9, np.nan)

# replace NaN with 0 for snow depth
df["snow_depth"] = df["snow_depth"].fillna(0)

# do interpolation for the remaining NaNs
df = df.interpolate()

# standardize data
removed_mean = df.mean()
removed_std = df.std()
weather = (df - removed_mean) / removed_std

# ---------------------------------------- Load events data
print("loading events data...")

events = pd.read_csv("../barclays_events_preprocessed.tsv", sep="\t")
events.head()

events['start_time'] = pd.to_datetime(events['start_time'], format='%Y-%m-%d %H:%M')
events['date'] = events['start_time'].dt.strftime("%Y-%m-%d")
events = events[["date", "start_time", "title", "url", "description"]]

# ---------------------------------------- Load taxi data (and merge with others and detrend)
print("loading taxi data (and merging and detrending)...")

df = pd.read_csv("../pickups_barclays_center_0.003.csv")

df_sum = pd.DataFrame(df.groupby("date")["pickups"].sum())
df_sum["date"] = df_sum.index
df_sum.index = pd.to_datetime(df_sum.index, format='%Y-%m-%d %H:%M')
df_sum["dow"] = df_sum.index.weekday

# add events information
event_col = np.zeros((len(df_sum)))
late_event = np.zeros((len(df_sum)))
really_late_event = np.zeros((len(df_sum)))
event_desc_col = []
for i in range(len(df_sum)):
    if df_sum.iloc[i].date in events["date"].values:
        event_col[i] = 1
        event_descr = ""
        #         for e in events[events.date == df_sum.iloc[i].date]["title"]:
        #             event_descr += str(e) + " "
        for e in events[events.date == df_sum.iloc[i].date]["description"]:
            event_descr += str(e) + " "
        event_desc_col.append(event_descr)
        for e in events[events.date == df_sum.iloc[i].date]["start_time"]:
            if e.hour >= 20:
                late_event[i] = 1
            if e.hour >= 21:
                really_late_event[i] = 1
    else:
        event_desc_col.append("None")

df_sum["event"] = event_col
df_sum["late_event"] = late_event
df_sum["really_late_event"] = really_late_event
df_sum["event_desc"] = event_desc_col

df_sum["event_next_day"] = pd.Series(df_sum["event"]).shift(-1)
df_sum["late_event_next_day"] = pd.Series(df_sum["late_event"]).shift(-1)
df_sum["really_late_event_next_day"] = pd.Series(df_sum["really_late_event"]).shift(-1)
df_sum["event_next_day_desc"] = pd.Series(df_sum["event_desc"]).shift(-1)

# merge with weather data
df_sum = df_sum.join(weather, how='inner')
df_sum.head()

# keep only data after 2013
START_YEAR = 2013
df_sum = df_sum.loc[df_sum.index.year >= START_YEAR]
df_sum.head()

df_sum["year"] = df_sum.index.year

trend_mean = df_sum[df_sum.index.year < 2015].groupby(["dow"]).mean()["pickups"]

# trend_std = df_sum.groupby(["year"]).std()["pickups"]
trend_std = df_sum["pickups"].std()

# build vectors with trend to remove and std
trend = []
std = []
for ix, row in df_sum.iterrows():
    trend.append(trend_mean[row.dow])
    # std.append(trend_std[row.year])
    std.append(trend_std)

df_sum["trend"] = trend
df_sum["std"] = std

# detrend data
df_sum["detrended"] = (df_sum["pickups"] - df_sum["trend"]) / df_sum["std"]

# ---------------------------------------- Build lags and features
print("building lags...")

event_texts = pd.concat([pd.Series(df_sum["event_desc"]).shift(x) for x in range(NUM_LAGS - 1, -1, -1)],
                        axis=1).values
# lags = pd.concat([pd.Series(df_sum["detrended"]).shift(x) for x in range(0,NUM_LAGS)],axis=1).values
lags = pd.concat([pd.Series(df_sum["detrended"]).shift(x) for x in range(NUM_LAGS - 1, -1, -1)], axis=1).values

event_feats = np.concatenate([df_sum["event_next_day"].values[:, np.newaxis],
                              df_sum["late_event"].values[:, np.newaxis],
                              # df_sum["late_event_next_day"].values[:,np.newaxis],
                              df_sum["really_late_event"].values[:, np.newaxis],
                              df_sum["really_late_event_next_day"].values[:, np.newaxis]], axis=1)
lags_event_feats = pd.concat([pd.Series(df_sum["event_next_day"]).shift(x) for x in range(0, NUM_LAGS)], axis=1).values
event_texts = df_sum["event_next_day_desc"].values
weather_feats = df_sum[['min_temp', 'max_temp', 'wind_speed',
                        'wind_gust', 'visibility', 'pressure', 'precipitation',
                        'snow_depth', 'fog', 'rain_drizzle', 'snow_ice', 'thunder']].values

# preds = pd.Series(df_sum["detrended"]).shift(-1).as_matrix()
preds = pd.concat([pd.Series(df_sum["detrended"]).shift(x) for x in range(- 1, -NUM_PRED - 1, -1)],
                  axis=1).values
trends = df_sum["trend"].shift(-1).values
stds = df_sum["std"].shift(-1).values

lags = lags[NUM_LAGS:-NUM_PRED, :]
event_feats = event_feats[NUM_LAGS:-NUM_PRED, :]
event_texts = event_texts[NUM_LAGS:-NUM_PRED]
weather_feats = weather_feats[NUM_LAGS:-NUM_PRED, :]
preds = preds[NUM_LAGS:-NUM_PRED]
trends = trends[NUM_LAGS:-NUM_PRED]
stds = stds[NUM_LAGS:-NUM_PRED]


# ---------------------------------------- Train/test split
print("loading train/val/test split...")

i_train = 365*2-90 # 2013 and 2014
i_val = 365*2
i_test = -1 # 2015 and 2016 (everything else)

lags = np.expand_dims(lags, axis=2)
lags_train = lags[:i_train,:] # time series lags
event_feats_train = event_feats[:i_train,:] # event/no_event
event_texts_train = event_texts[:i_train] # event text descriptions
weather_feats_train = weather_feats[:i_train,sel ] # weather data
y_train = preds[:i_train] # target values

lags_val = lags[i_train:i_val,:] # time series lags
event_feats_val = event_feats[i_train:i_val,:] # event/no_event
event_texts_val = event_texts[i_train:i_val] # event text descriptions
weather_feats_val = weather_feats[i_train:i_val,sel ] # weather data
y_val = preds[i_train:i_val] # target values

lags_test = lags[i_val:i_test,:]
event_feats_test = event_feats[i_val:i_test,:]
event_texts_test = event_texts[i_val:i_test]
weather_feats_test = weather_feats[i_val:i_test,sel ]
y_test = preds[i_val:i_test]
trend_test = trends[i_val:i_test]
std_test = stds[i_val:i_test]


def compute_error(trues, predicted):
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    rrse = np.sqrt(np.sum((predicted - trues)**2) / np.sum((trues - np.mean(trues))**2))
    mape = np.mean(np.abs((predicted - trues) / trues)) * 100
    r2 = max(0, 1 - np.sum((predicted - trues)**2) / np.sum((trues - np.mean(trues))**2))
    return corr, mae, rae, rmse, rrse, mape, r2


def add_weight_decay(model, lags_decay=0.2, weather_decay=0.1, event_decay=0.02, b_decay=0.001,text_decay =0.1, merge_decay=0.1,
                     other_decay=0.02):
    lags_params = []
    weather_params = []
    event_params = []
    b_params = []
    # text_params = []
    merge_parmas = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            b_params.append(param)
        else:
            if name.startswith('lags'):
                lags_params.append(param)
            elif name.startswith('weather'):
                weather_params.append(param)
            elif name.startswith('event'):
                event_params.append(param)
            # elif name.startswith('text'):
            #     text_params.append(param)
            elif name.startswith('layer'):
                merge_parmas.append(param)
            else:
                other_params.append(param)

    return [
        {'params': b_params, 'weight_decay': b_decay},
        {'params': lags_params, 'weight_decay': lags_decay},
        {'params': weather_params, 'weight_decay': weather_decay},
        {'params': event_params, 'weight_decay': event_decay},
        # {'params': text_params, 'weight_decay': text_decay},
        {'params': merge_parmas, 'weight_decay': merge_decay},
        {'params': other_params, 'weight_decay': other_decay}]


class Encoder_decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_reg=1, num_pred=3, num_lags=9):
        super(Encoder_decoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_pred = num_pred
        self.active = F.tanh
        self.dropout5 = nn.Dropout(0.5)

        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        self.decoder_regression = nn.Linear(hidden_dim, n_reg)

    #         self.regression = nn.Linear(hidden_dim , n_reg)

    def forward(self, inputs, extract_feature=False):

        encoder_out, (encoder_hn, encoder_cn) = self.encoder(inputs)
        if extract_feature:
            return encoder_out[:, -1, :]
        else:
            for i in range(self.num_pred):
                if i == 0:
                    last_input = inputs[:, -1, :]
                    last_input = torch.reshape(last_input, [-1, 1, self.in_dim])
                    decoder_out, (decoder_hn, decoder_cn) = self.decoder(last_input, (encoder_hn, encoder_cn))
                    decoder_out = self.dropout5(decoder_out)
                    pred = self.decoder_regression(decoder_out)

                    pred_list = pred
                else:
                    decoder_out, (decoder_hn, decoder_cn) = self.decoder(pred, (decoder_hn, decoder_cn))
                    decoder_out = self.dropout5(decoder_out)
                    pred = self.decoder_regression(decoder_out)
                    pred_list = torch.cat((pred_list, pred), 1)
        pred_list = torch.reshape(pred_list, [-1, self.num_pred])

        return pred_list


class Regression(nn.Module):
    def __init__(self, hidden_dim, n_reg, pred_train_path):
        super(Regression, self).__init__()
        self.active = F.tanh
        self.dropout5 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.pred_train_model = torch.load(pred_train_path)
        #         self.weather_layer = nn.Linear(7 , hidden_dim)
        self.weather_norm7 = nn.BatchNorm1d(7)
        self.weather_hidden200 = nn.Linear(7, hidden_dim)
        self.weather_norm200 = nn.BatchNorm1d(hidden_dim)
        self.weather_hidden50 = nn.Linear(hidden_dim, hidden_dim)
        self.weather_norm50 = nn.BatchNorm1d(hidden_dim)

        self.event_hidden = nn.Linear(4, hidden_dim)

        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)

        self.regression = nn.Linear(hidden_dim * 3, n_reg)

    def forward(self, lags, weather, event):
        with torch.no_grad():
            feature = self.pred_train_model(lags, extract_feature=True)
        lags_feature = self.hidden_layer(feature)

        #         weather_feature = self.weather_layer(weather)
        weather_feature = self.weather_hidden200(weather)
        weather_feature = self.active(weather_feature)
        weather_feature = self.dropout5(weather_feature)
        weather_feature = self.weather_hidden50(weather_feature)
        weather_feature = self.active(weather_feature)
        weather_feature = self.dropout4(weather_feature)

        event_feature = self.event_hidden(event)

        feature = torch.cat((lags_feature, weather_feature, event_feature), 1)
        feature = self.dropout5(feature)
        feature = self.active(feature)
        out = self.regression(feature)
        return out


def model_test(data_loader, model, is_pre_train=False):
    '''
    使用验证集或测试集测试模型
    '''
    pred_Y = []
    test_Y = []

    for step, (test_x_batch, test_weather_batch, test_event_batch, test_y_batch) in enumerate(data_loader):
        test_y_batch = test_y_batch.numpy()
        test_x_batch = test_x_batch.cuda(DEVICE)
        test_weather_batch = test_weather_batch.cuda(DEVICE)
        test_event_batch = test_event_batch.cuda(DEVICE)

        if is_pre_train == False:
            pred_y_batch = model(test_x_batch, test_weather_batch, test_event_batch)
        else:
            pred_y_batch = model(test_x_batch)

        pred_y_batch = pred_y_batch.cpu().data.numpy()
        if is_pre_train == False:
            pred_y_batch = pred_y_batch[:, 0]
            test_y_batch = test_y_batch[:, 0]
        if pred_Y == [] and test_Y == []:
            pred_Y = pred_y_batch
            test_Y = test_y_batch
        else:
            pred_Y = np.concatenate((pred_Y, pred_y_batch), axis=0)
            test_Y = np.concatenate((test_Y, test_y_batch), axis=0)

    val_mse = np.mean(np.square(test_Y - pred_Y))
    #     print('val_mse:',val_mse)
    return pred_Y, val_mse


import torch.optim as optim
from torch.utils import data

# Use PyTorch's DataLoader and the collate function
# defined before.

train_set = data.TensorDataset(torch.FloatTensor(lags_train), torch.FloatTensor(weather_feats_train),
                               torch.FloatTensor(event_feats_train), torch.FloatTensor(y_train))
test_set = data.TensorDataset(torch.FloatTensor(lags_test), torch.FloatTensor(weather_feats_test),
                              torch.FloatTensor(event_feats_test), torch.FloatTensor(y_test))
val_set = data.TensorDataset(torch.FloatTensor(lags_val), torch.FloatTensor(weather_feats_val),
                             torch.FloatTensor(event_feats_val), torch.FloatTensor(y_val))

train_data_loader = data.DataLoader(train_set, batch_size=batch_size,
                                    shuffle=True)
test_data_loader = data.DataLoader(test_set, batch_size=batch_size,
                                   shuffle=False)
val_data_loader = data.DataLoader(val_set, batch_size=batch_size,
                                  shuffle=False)


def pred_train(in_dim=1, hidden_dim=64, n_reg=1,num_pred=3,num_lags=9):
    # Create model
    model = Encoder_decoder(in_dim=1, hidden_dim=64, n_reg=1,num_pred=3,num_lags=9)
    model.to(DEVICE)
    loss_func = nn.MSELoss()

#     parm = add_weight_decay(model, lags_decay=lags_decay, weather_decay=0, event_decay=0, b_decay=0, merge_decay=0,
#                             other_decay=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    min_val_mse = 1
    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for iter, (x,_,_, y) in enumerate(train_data_loader):
            x = x.cuda(DEVICE)
            y = y.cuda(DEVICE)
            prediction = model(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)

        model.eval()
        pred_Y, val_mse = model_test(val_data_loader, model,is_pre_train = True)
#         print(epoch,val_mse)
        if val_mse < min_val_mse:
            min_val_mse = val_mse
            torch.save(model, 'pred_train_barclays_lags_weather_event.pkl')

    return 'pred_train_barclays_lags_weather_event.pkl'


def run_model(pred_train_path,hidden_dim=64, n_reg=1):
    # Create model
    model = Regression(hidden_dim, n_reg,pred_train_path)
    model.to(DEVICE)
    loss_func = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    min_val_mse = 1
    for epoch in range(60):
        model.train()
        epoch_loss = 0
        for iter, (lags,weather,event, y) in enumerate(train_data_loader):
            lags = lags.cuda(DEVICE)
            weather = weather.cuda(DEVICE)
            event = event.cuda(DEVICE)
            y = y.cuda(DEVICE)
            y = y[:,0]
            prediction = model(lags,weather,event)
            loss = loss_func(prediction.squeeze(-1), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)

        model.eval()
        pred_Y, val_mse = model_test(val_data_loader, model)
#         print(epoch, val_mse)
        if val_mse < min_val_mse:
            min_val_mse = val_mse
            torch.save(model, 'barclays_lags_weather_event.pkl')

    model = torch.load('barclays_lags_weather_event.pkl')
    model.eval()
    pred_Y, val_mse = model_test(test_data_loader, model)

    preds_lstm = pred_Y * std_test + trend_test
    y_true = y_test[:,0] * std_test + trend_test
    corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lstm)
    print(
        '{corr}\t{mae}\t{rae}\t{rmse}\t{rrse}\t{mape}\t{r2}\n'.format(corr=corr, mae=mae, rae=rae, rmse=rmse, rrse=rrse,
                                                                      mape=mape, r2=r2))
    with open(result_file, 'a') as f:
        f.write('{corr}\t{mae}\t{rae}\t{rmse}\t{rrse}\t{mape}\t{r2}\n'.format(corr=corr, mae=mae, rae=rae, rmse=rmse,
                                                                              rrse=rrse, mape=mape, r2=r2))

    return r2



if __name__ == '__main__':

    with open(result_file,'w') as f:
        f.write('corr\tmae\trae\trmse\trrse\tmape\tr2\n')

    for i in range(30):
        pred_train_path = pred_train()
        run_model(pred_train_path)