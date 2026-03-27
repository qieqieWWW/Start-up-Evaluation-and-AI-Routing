#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#SDNIP


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

# 最优配置
FEATURE_COMB = "comb_full"
FEATURE_WEIGHT = "w_normal"

# 固定配置
FEATURE_COMBINATION = {"comb_full": [0,1,2,3,4,5]}
FEATURE_WEIGHT = {"w_normal": [1,1,1,1,1,1]}

def load_data(feature_comb_name, feature_weight_name):
    with open("processed_data.csv", "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = f.readlines()
    X_all = []
    y_all = []
    comb_idx = FEATURE_COMBINATION[feature_comb_name]
    weight = FEATURE_WEIGHT[feature_weight_name]
    for line in lines[1:]:
        line = line.strip()
        if not line: continue
        parts = line.split(",")
        if len(parts) < 11: continue
        try:
            raw = [float(parts[5]), float(parts[6]), float(parts[7]),
                   float(parts[8]), float(parts[9]), float(parts[10])]
            feat = [raw[i] * weight[i] for i in comb_idx]
            label = int(float(parts[4]))
            X_all.append(feat)
            y_all.append(label)
        except:
            continue
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SDNIP最优模型
class SDNIP_NoThreshold:
    def fit(self, X, y):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
    def predict(self, X):
        return (np.mean(X, axis=1) > 0.5).astype(int)

# 运行最优版本
def run_sdnip_best():
    X_train, X_test, y_train, y_test = load_data(FEATURE_COMB, FEATURE_WEIGHT)
    model = SDNIP_NoThreshold()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    pre = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    res = {
        "model": "SDNIP",
        "feature_comb": FEATURE_COMB,
        "feature_weight": FEATURE_WEIGHT,
        "accuracy": round(acc,4),
        "recall": round(rec,4),
        "precision": round(pre,4),
        "f1": round(f1,4)
    }
    print("✅ SDNIP最优版本结果：", res)
    return res

if __name__ == "__main__":
    run_sdnip_best()


# In[ ]:


#MultiAgent


# In[ ]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

# 最优配置
FEATURE_COMB = "comb_full"
FEATURE_WEIGHT = "w_risk_5x"
HP_NAME = "hp_low_drop"

FEATURE_COMBINATION = {"comb_full": [0,1,2,3,4,5]}
FEATURE_WEIGHT = {"w_risk_5x": [1,1,1,5,1,1]}
MODEL_PARAMS = {"hp_low_drop": {"dropout":0.1, "lr":0.001}}

TRAIN_CONFIG = {'epochs': 12, 'lr': 0.001, 'batch_size': 256}

def load_data(feature_comb_name, feature_weight_name):
    with open("processed_data.csv", "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = f.readlines()
    X_all = []
    y_all = []
    comb_idx = FEATURE_COMBINATION[feature_comb_name]
    weight = FEATURE_WEIGHT[feature_weight_name]
    for line in lines[1:]:
        line = line.strip()
        if not line: continue
        parts = line.split(",")
        if len(parts) < 11: continue
        try:
            raw = [float(parts[5]), float(parts[6]), float(parts[7]),
                   float(parts[8]), float(parts[9]), float(parts[10])]
            feat = [raw[i] * weight[i] for i in comb_idx]
            label = int(float(parts[4]))
            X_all.append(feat)
            y_all.append(label)
        except:
            continue
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 模型结构
class GCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(F.relu(self.bn(self.fc(x))))

class MultiAgentGCN(nn.Module):
    def __init__(self, in_dim, drop=0.3):
        super().__init__()
        self.layers = nn.Sequential(GCNBlock(in_dim, 32, drop), GCNBlock(32, 16, drop))
        self.out = nn.Linear(16, 2)
    def forward(self, x):
        return self.out(self.layers(x))

# 运行最优版本
def run_multi_best():
    X_train, X_test, y_train, y_test = load_data(FEATURE_COMB, FEATURE_WEIGHT)
    hp = MODEL_PARAMS[HP_NAME]
    model = MultiAgentGCN(in_dim=X_train.shape[1], drop=hp["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.long).to(device)

    model.train()
    for e in range(TRAIN_CONFIG["epochs"]):
        for i in range(0, len(Xt), TRAIN_CONFIG["batch_size"]):
            bx = Xt[i:i+TRAIN_CONFIG["batch_size"]]
            by = yt[i:i+TRAIN_CONFIG["batch_size"]]
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            opt.zero_grad()

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = torch.argmax(model(xt), dim=1).cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    pre = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    res = {
        "model": "MultiAgent",
        "feature_comb": FEATURE_COMB,
        "feature_weight": FEATURE_WEIGHT,
        "hp": HP_NAME,
        "accuracy": round(acc,4),
        "recall": round(rec,4),
        "precision": round(pre,4),
        "f1": round(f1,4)
    }
    print("✅ MultiAgent最优版本结果：", res)
    return res

if __name__ == "__main__":
    run_multi_best()


# In[ ]:


#Capsule


# In[ ]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

# 最优配置
FEATURE_COMB = "comb_full"
FEATURE_WEIGHT = "w_risk_10x"
HP_NAME = "hp_normal"

FEATURE_COMBINATION = {"comb_full": [0,1,2,3,4,5]}
FEATURE_WEIGHT = {"w_risk_10x": [1,1,1,10,1,1]}
MODEL_PARAMS = {"hp_normal": {"dropout":0.3, "lr":0.001}}

TRAIN_CONFIG = {'epochs': 12, 'lr': 0.001, 'batch_size': 256}

def load_data(feature_comb_name, feature_weight_name):
    with open("processed_data.csv", "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = f.readlines()
    X_all = []
    y_all = []
    comb_idx = FEATURE_COMBINATION[feature_comb_name]
    weight = FEATURE_WEIGHT[feature_weight_name]
    for line in lines[1:]:
        line = line.strip()
        if not line: continue
        parts = line.split(",")
        if len(parts) < 11: continue
        try:
            raw = [float(parts[5]), float(parts[6]), float(parts[7]),
                   float(parts[8]), float(parts[9]), float(parts[10])]
            feat = [raw[i] * weight[i] for i in comb_idx]
            label = int(float(parts[4]))
            X_all.append(feat)
            y_all.append(label)
        except:
            continue
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 胶囊模型
class SimpleCapsule(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 16)
        self.out = nn.Linear(16, 2)
    def forward(self, x):
        return self.out(torch.relu(self.fc(x)))

# 运行最优版本
def run_capsule_best():
    X_train, X_test, y_train, y_test = load_data(FEATURE_COMB, FEATURE_WEIGHT)
    hp = MODEL_PARAMS[HP_NAME]
    model = SimpleCapsule(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.long).to(device)

    model.train()
    for e in range(TRAIN_CONFIG["epochs"]):
        for i in range(0, len(Xt), TRAIN_CONFIG["batch_size"]):
            bx = Xt[i:i+TRAIN_CONFIG["batch_size"]]
            by = yt[i:i+TRAIN_CONFIG["batch_size"]]
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            opt.zero_grad()

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = torch.argmax(model(xt), dim=1).cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    pre = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    res = {
        "model": "Capsule",
        "feature_comb": FEATURE_COMB,
        "feature_weight": FEATURE_WEIGHT,
        "hp": HP_NAME,
        "accuracy": round(acc,4),
        "recall": round(rec,4),
        "precision": round(pre,4),
        "f1": round(f1,4)
    }
    print("✅ Capsule最优版本结果：", res)
    return res

if __name__ == "__main__":
    run_capsule_best()

