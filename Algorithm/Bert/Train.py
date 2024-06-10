import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

def plot_text_length_distribution(df):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    length_counts = df['text'].apply(len).value_counts().sort_index()

    # 绘制直方图
    plt.hist(length_counts.index, bins=len(length_counts), weights=length_counts.values)
    plt.xlabel('文本长度')
    plt.ylabel('频数')
    plt.title('字符串长度分布直方图')
    plt.show()

class NewsClassifier:
    def __init__(self, data_paths, label_path, model_path, save_path, seed=1999):
        self.data_paths = data_paths
        self.label_path = label_path
        self.model_path = model_path
        self.save_path = save_path
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.labels = self._load_labels()
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_dataset = None
        self.dev_dataset = None

        self.setup_seed()
        self.model = BertClassifier(self.labels, self.model_path).to(self.device)

        train_df = self.load_data(self.data_paths['train'], 200)
        dev_df = self.load_data(self.data_paths['dev'], 100)
        self.train_dataset = self.create_dataset(train_df)
        self.dev_dataset = self.create_dataset(dev_df)
        # plot_text_length_distribution(train_df)

    def _load_labels(self):
        with open(self.label_path, 'r') as f:
            return [row.strip() for row in f.readlines()]

    def setup_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def load_data(self, path, limit=None):
        df = pd.read_csv(path, header=None)
        if limit:
            df = df.head(limit)
        df.columns = ['text', 'label']
        df.dropna(how='all', inplace=True)
        df = df.dropna(subset=["label"])
        df["text"] = df["text"].astype(object)
        df["label"] = df["label"].astype('int64')
        return df

    def create_dataset(self, df):
        texts = [self.tokenizer(text, padding='max_length', max_length=35, truncation=True, return_tensors="pt") for text in df["text"]]
        labels = df['label'].values
        return MyDataset(texts, labels)

    def train(self, epochs, batch_size, lr):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(self.dev_dataset, batch_size=batch_size)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_dev_acc = 0
        for epoch in range(epochs):
            print("Epoch = ", epoch + 1)
            self._train_epoch(train_loader)
            val_acc = self._evaluate_epoch(dev_loader)
            if val_acc > best_dev_acc:
                best_dev_acc = val_acc
                self._save_model('best_model.pt')

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_acc_train = 0
        for inputs, labels in tqdm(train_loader):
            input_ids = inputs['input_ids'].squeeze(1).to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            acc = (outputs.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc
            total_loss += loss.item()
        print(f'''Train Loss: {total_loss / len(self.train_dataset): .3f} 
          | Train Accuracy: {total_acc_train / len(self.train_dataset): .3f}''')

    def _evaluate_epoch(self, dev_loader):
        self.model.eval()
        total_acc = 0
        total_loss_val = 0
        with torch.no_grad():
            for inputs, labels in dev_loader:
                input_ids = inputs['input_ids'].squeeze(1).to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids, attention_mask)

                batch_loss = self.criterion(outputs, labels)
                acc = (outputs.argmax(dim=1) == labels).sum().item()
                total_acc += acc
                total_loss_val += batch_loss.item()
        print(f'''Val Loss: {total_loss_val / len(self.dev_dataset): .3f} 
          | Val Accuracy: {total_acc / len(self.dev_dataset): .3f}''')
        return total_acc / len(dev_loader)

    def _save_model(self, save_name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, save_name))

    def evaluate(self, test_df):
        self.model.eval()
        test_dataset = self.create_dataset(test_df)
        test_loader = DataLoader(test_dataset, batch_size=1)
        total_acc = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                input_ids = inputs['input_ids'].squeeze(1).to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids, attention_mask)
                acc = (outputs.argmax(dim=1) == labels).sum().item()
                total_acc += acc
        print(f'Test Accuracy: {total_acc / len(test_dataset)}')

    def predict(self, text):
        self.model.eval()
        input_ids = self.tokenizer(text, padding='max_length', max_length=35, truncation=True, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        attention_mask = self.tokenizer(text, padding='max_length', max_length=35, truncation=True, return_tensors="pt").attention_mask
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids.squeeze(1), attention_mask)
        pred = self.labels[outputs.argmax(dim=1).item()]
        return pred

class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

class BertClassifier(nn.Module):
    def __init__(self, labels, model_path):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.5)
        # 确保输出层的大小与标签的数量相匹配
        self.linear = nn.Linear(self.bert.config.hidden_size, len(labels))
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        outputs = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        pooled_output = outputs[1]
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_output = self.relu(linear_output)
        return final_output

if __name__ == "__main__":
    data_paths = {
        'train': './Data/THUCNewsPart/train.csv',
        'dev': './Data/THUCNewsPart/dev.csv',
        'test': './Data/THUCNewsPart/test.csv'
    }
    news_classifier = NewsClassifier(data_paths = data_paths, label_path = './Data/THUCNewsPart/class.txt', model_path = './PretraionModel', save_path = './Data/THUCNewsPart/')
    news_classifier.train(epochs=5, batch_size=1, lr=1e-5)

    # 评估测试集
    test_df = news_classifier.load_data(data_paths['test'], 100)
    news_classifier.model.load_state_dict(torch.load(os.path.join(news_classifier.save_path, 'best_model.pt')))
    news_classifier.evaluate(test_df)

    # 预测
    while True:
        text = input('是否和贷款相关：')
        print(news_classifier.predict(text))