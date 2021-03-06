import os
DEVICE = 'cuda'

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import datasets, linear_model
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
from matplotlib import cm

import statsmodels.formula.api as smf
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from itertools import combinations
import torch.optim as optim
from torch.utils import data
from gensim.models import KeyedVectors

# ---------------------------------------- GLOBAL PARAMETERS

NUM_LAGS = 10
# sel = [0,2,4,5,7,8,9] # weather features to use
sel = [5, 7]
result_file = 'termial5_lags_result.txt'

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

events = pd.read_csv("../terminal5_events_preprocessed.tsv", sep="\t")
events.head()

events['start_time'] = pd.to_datetime(events['start_time'], format='%Y-%m-%d %H:%M')
events['date'] = events['start_time'].dt.strftime("%Y-%m-%d")
events = events[["date", "start_time", "title", "url", "description"]]

# ---------------------------------------- Load taxi data (and merge with others and detrend)
print("loading taxi data (and merging and detrending)...")
df = pd.read_csv("../pickups_terminal_5_0.003.csv")
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
df_sum.shape

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

lags_event_feats = []
event_feat_list = ['event_next_day', 'late_event', 'really_late_event', 'really_late_event_next_day']
for i in range(len(event_feat_list)):
    event_feats = pd.concat([pd.Series(df_sum[event_feat_list[i]]).shift(x) for x in range(NUM_LAGS - 1, -1, -1)],
                            axis=1).values
    event_feats = event_feats[:, :, np.newaxis]
    if i == 0:
        lags_event_feats = event_feats
    else:
        lags_event_feats = np.concatenate((lags_event_feats, event_feats), axis=2)

lags_weather_feats = []
weather_feat_list = ['min_temp', 'max_temp', 'wind_speed', 'wind_gust', 'visibility', 'pressure', 'precipitation',
                     'snow_depth', 'fog', 'rain_drizzle', 'snow_ice', 'thunder']
for i in range(len(weather_feat_list)):
    weather_feat = pd.concat([pd.Series(df_sum[weather_feat_list[i]]).shift(x) for x in range(NUM_LAGS - 1, -1, -1)],
                             axis=1).values
    weather_feat = weather_feat[:, :, np.newaxis]
    if i == 0:
        lags_weather_feats = weather_feat
    else:
        lags_weather_feats = np.concatenate((lags_weather_feats, weather_feat), axis=2)

preds = pd.Series(df_sum["detrended"]).shift(-1).values
trends = df_sum["trend"].shift(-1).values
stds = df_sum["std"].shift(-1).values

lags = lags[NUM_LAGS:-1, :]
lags_event_feats = lags_event_feats[NUM_LAGS:-1, :]
event_texts = event_texts[NUM_LAGS:-1]
lags_weather_feats = lags_weather_feats[NUM_LAGS:-1, :]
preds = preds[NUM_LAGS:-1]
trends = trends[NUM_LAGS:-1]
stds = stds[NUM_LAGS:-1]



# ---------------------------------------- Train/test split
print("loading train/val/test split...")

i_train = 365*2 # 2013 and 2014
i_val = 365*3
i_test = -1 # 2015 and 2016 (everything else)

lags_train = lags[:i_train,:] # time series lags
event_feats_train = lags_event_feats[:i_train,:] # event/no_event
event_texts_train = event_texts[:i_train] # event text descriptions
weather_feats_train = lags_weather_feats[:i_train,:,sel] # weather data
y_train = preds[:i_train] # target values

lags_val = lags[i_train:i_val,:] # time series lags
event_feats_val = lags_event_feats[i_train:i_val,:] # event/no_event
event_texts_val = event_texts[i_train:i_val] # event text descriptions
weather_feats_val = lags_weather_feats[i_train:i_val,:,sel] # weather data
y_val = preds[i_train:i_val] # target values

lags_test = lags[i_val:i_test,:]
event_feats_test = lags_event_feats[i_val:i_test,:]
event_texts_test = event_texts[i_val:i_test]
weather_feats_test = lags_weather_feats[i_val:i_test,:,sel]
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


import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from itertools import combinations


def get_batch_graph(batch_size, max_len, windows):
    batch_graph = []
    for i in range(batch_size):
        g = dgl.DGLGraph()
        g.add_nodes(max_len)
        # A couple edges one-by-one
        for j in range(windows, max_len + 1):
            t = range(j - windows, j)
            edges = np.array(list(combinations(t, 2)))
            for edge in edges[0:-1]:
                g.add_edges(edge[0], edge[1])
        #                 g.add_edge(edge[1], edge[0])
        g.add_edge(max_len - 2, max_len - 1)
        g.add_edge(max_len - 1, 0)
        #         g.add_edge(0,max_len-1)
        batch_graph.append(g)

    return batch_graph


batch_size = 32
batch_graph = get_batch_graph(batch_size, NUM_LAGS, 3)


def get_next_batch(batch_size, batch_graph=batch_graph):
    batch_graph = batch_graph[0:batch_size]
    bg = dgl.batch(batch_graph, edge_attrs=None)
    bg = bg.to(torch.device(DEVICE))
    return bg


# import networkx as nx
# import matplotlib.pyplot as plt
# g_nx = batch_graph[0]
# nx.draw(g_nx.to_networkx(), with_labels=True)
# plt.show()


class MGN(nn.Module):
    #reference: https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html?highlight=gcn
    def __init__(self, hidden_dim, activation):
        super(MGN, self).__init__()
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.merge_linner = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation



    def message_func(self, edges):
        return {'h': edges.src['h']}

    def reduce_func(self, nodes):
        accum = torch.sum(nodes.mailbox['h'], 1)
        accum = self.merge_linner(accum)
        return {'h': accum}

    def forward(self, bg):
        #         bg.apply_edges(self.edge_attention)
        bg.update_all(self.message_func, self.reduce_func)
        return bg


import torch.nn.functional as F


class Regression(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_reg):
        super(Regression, self).__init__()
        self.active = F.tanh
        self.MGN = MGN(hidden_dim, self.active)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        self.tanh = F.tanh
        self.dropout5 = nn.Dropout(0.3)
        self.dropout = nn.Dropout(0.2)
        self.lags_norm10 = nn.BatchNorm1d(in_dim)
        self.lags_hidden200 = nn.Linear(in_dim, hidden_dim)
        self.lags_norm200 = nn.BatchNorm1d(hidden_dim)
        self.lags_hidden50 = nn.Linear(hidden_dim, hidden_dim)
        self.lags_norm50 = nn.BatchNorm1d(hidden_dim)

        self.regression = nn.Linear(hidden_dim * NUM_LAGS, n_reg)

    def forward(self, inputs):
        bs = inputs.shape[0]
        h = torch.reshape(inputs, (-1, 1))

        # h = self.lags_norm10(h)
        h = self.lags_hidden200(h)
        h = self.tanh(h)
        h = self.dropout5(h)
        h = self.lags_norm200(h)
        h = self.lags_hidden50(h)
        h = self.tanh(h)
        h = self.dropout(h)

        bg = get_next_batch(bs)
        bg.ndata['h'] = h
        # For undirected graphs, in_degree is the same as
        # out_degree.
        bg = self.MGN(bg)
        hg = torch.reshape(bg.ndata['h'], (bs, NUM_LAGS, -1))
        out, hidden = self.gru(hg)

        hg = torch.reshape(out, (bs, -1))

        return self.regression(hg)


def model_test(data_loader,model):
    '''
    Test models with validation or test sets
    '''
    pred_Y = []
    test_Y = []

    for step,(test_x_batch,test_y_batch) in enumerate(data_loader):

        test_y_batch = test_y_batch.numpy()
        test_x_batch = test_x_batch.cuda(DEVICE)
        pred_y_batch = model(test_x_batch)

        pred_y_batch = pred_y_batch.cpu().data.numpy()
        pred_y_batch = pred_y_batch[:,0]
        pred_Y = np.concatenate((pred_Y,pred_y_batch), axis=0)
        test_Y = np.concatenate((test_Y,test_y_batch), axis=0)

    val_mse = np.mean(np.square(test_Y - pred_Y))
#     print('val_mse:',val_mse)
    return pred_Y, val_mse

import torch.optim as optim
from torch.utils import data


# Use PyTorch's DataLoader and the collate function
# defined before.
train_set = data.TensorDataset(torch.FloatTensor(lags_train), torch.FloatTensor(y_train))
test_set = data.TensorDataset(torch.FloatTensor(lags_test), torch.FloatTensor(y_test))
val_set = data.TensorDataset(torch.FloatTensor(lags_val), torch.FloatTensor(y_val))

train_data_loader = data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=True)
test_data_loader = data.DataLoader(test_set, batch_size=batch_size,
                                        shuffle=False)
val_data_loader = data.DataLoader(val_set, batch_size=batch_size,
                                        shuffle=False)

def run_model(hidden_dim=128, lags_decay=0.05):
    # Create model
    model = Regression(1, hidden_dim, 1)
    model.to(DEVICE)
    loss_func = nn.MSELoss()

    parm=add_weight_decay(model, lags_decay=lags_decay,weather_decay = 0, event_decay = 0, b_decay=0,merge_decay =0,other_decay = 0)
    optimizer = optim.Adam(parm, lr=0.001)


    min_val_mse = 1
    for epoch in range(160):
        model.train()
        epoch_loss = 0
        for iter, (x, y) in enumerate(train_data_loader):
            x = x.cuda(DEVICE)
            y = y.cuda(DEVICE)
            prediction = model(x)
            loss = loss_func(prediction.squeeze(-1), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)

        model.eval()
        pred_Y, val_mse = model_test(val_data_loader,model)
        if val_mse<min_val_mse:
            min_val_mse = val_mse
            torch.save(model, 'terminal_lags.pkl')

    model = torch.load('terminal_lags.pkl')
    model.eval()
    pred_Y, val_mse = model_test(test_data_loader,model)

    preds_lstm = pred_Y * std_test + trend_test
    y_true =  y_test * std_test + trend_test
    corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lstm)
    print('{corr}\t{mae}\t{rae}\t{rmse}\t{rrse}\t{mape}\t{r2}\n'.format(corr = corr,mae = mae,rae = rae,rmse = rmse,rrse = rrse,mape=mape,r2 = r2))
    with open(result_file,'a') as f:
        f.write('{corr}\t{mae}\t{rae}\t{rmse}\t{rrse}\t{mape}\t{r2}\n'.format(corr = corr,mae = mae,rae = rae,rmse = rmse,rrse = rrse,mape=mape,r2 = r2))

    return r2

if __name__ == '__main__':
    with open(result_file, 'w') as f:
        f.write('corr\tmae\trae\trmse\trrse\tmape\tr2\n')

    for i in range(30):
        run_model()


