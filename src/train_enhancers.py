import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn import metrics
import os
import csv

############# HYPER-PARAMETERS ############
MY_RANDOM_STATE = 5
torch.manual_seed(MY_RANDOM_STATE)
NUMBER_EPOCHS = 20
LEARNING_RATE = 1e-4
SAMPLE_LENGTH = 200
AVGPOOL1D_KERNEL_SIZE = 4
CONV1D_KERNEL_SIZE = 3
BRANCH_BLOCK1_OUT = 8
BRANCH_BLOCK2_OUT = 16
FULLY_CONNECTED_LAYER_SIZE = 256

MODEL_DIR = 'full_training_model'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
###########################################

def kmer_to_index(kmer, k):
    value = 0
    for char in kmer:
        if char == 'A':
            val = 0
        elif char == 'C':
            val = 1
        elif char == 'G':
            val = 2
        elif char == 'T':
            val = 3
        else:
            raise ValueError(f"Invalid nucleotide '{char}' in k-mer {kmer}")
        value = value * 4 + val
    return value

def load_text_file(file_text):
    with open(file_text) as f:
        lines = f.readlines()
        my_data = [line.strip().upper() for line in lines[1::2]]
        return my_data

class EnhancerDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        label = self.Y[index]
        sample = self.X[index]
        padded_sample = sample.ljust(SAMPLE_LENGTH + 3, 'A')
        k1_indices = []
        k2_indices = []
        k3_indices = []
        k4_indices = []
        for i in range(SAMPLE_LENGTH):
            k1 = padded_sample[i]
            k1_idx = kmer_to_index(k1, 1)
            k1_indices.append(k1_idx)
            k2 = padded_sample[i:i+2]
            k2_idx = kmer_to_index(k2, 2)
            k2_indices.append(k2_idx)
            k3 = padded_sample[i:i+3]
            k3_idx = kmer_to_index(k3, 3)
            k3_indices.append(k3_idx)
            k4 = padded_sample[i:i+4]
            k4_idx = kmer_to_index(k4, 4)
            k4_indices.append(k4_idx)
        input_tensor = torch.stack([
            torch.LongTensor(k1_indices),
            torch.LongTensor(k2_indices),
            torch.LongTensor(k3_indices),
            torch.LongTensor(k4_indices)
        ], dim=0)
        return input_tensor.long(), label

    def __len__(self):
        return len(self.X)

class ConvBranch(nn.Module):
    def __init__(self, in_channels, block1_out, block2_out):
        super(ConvBranch, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, block1_out, kernel_size=CONV1D_KERNEL_SIZE, padding=1),
            nn.BatchNorm1d(block1_out),
            nn.ReLU(),
            nn.Conv1d(block1_out, block1_out, kernel_size=CONV1D_KERNEL_SIZE, padding=1),
            nn.BatchNorm1d(block1_out),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(AVGPOOL1D_KERNEL_SIZE)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(block1_out, block2_out, kernel_size=CONV1D_KERNEL_SIZE, padding=1),
            nn.BatchNorm1d(block2_out),
            nn.ReLU(),
            nn.Conv1d(block2_out, block2_out, kernel_size=CONV1D_KERNEL_SIZE, padding=1),
            nn.BatchNorm1d(block2_out),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(AVGPOOL1D_KERNEL_SIZE)
        )
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out

class EnhancerCnnModel(nn.Module):
    def __init__(self):
        super(EnhancerCnnModel, self).__init__()
        self.embedding_k1 = nn.Embedding(4, 128)
        self.embedding_k2 = nn.Embedding(16, 128)
        self.embedding_k3 = nn.Embedding(64, 128)
        self.embedding_k4 = nn.Embedding(256, 128)
        self.branch_k1 = ConvBranch(128, BRANCH_BLOCK1_OUT, BRANCH_BLOCK2_OUT)
        self.branch_k2 = ConvBranch(128, BRANCH_BLOCK1_OUT, BRANCH_BLOCK2_OUT)
        self.branch_k3 = ConvBranch(128, BRANCH_BLOCK1_OUT, BRANCH_BLOCK2_OUT)
        self.branch_k4 = ConvBranch(128, BRANCH_BLOCK1_OUT, BRANCH_BLOCK2_OUT)
        self.fc = nn.Linear(768, FULLY_CONNECTED_LAYER_SIZE)
        self.out = nn.Linear(FULLY_CONNECTED_LAYER_SIZE, 1)
        self.criterion = nn.BCELoss()

        # L2 regularization added via weight_decay in Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # L2 regularization

    def forward(self, inputs):
        batch_size = inputs.size(0)
        k1 = inputs[:, 0, :].long()
        k2 = inputs[:, 1, :].long()
        k3 = inputs[:, 2, :].long()
        k4 = inputs[:, 3, :].long()
        emb1 = self.embedding_k1(k1).permute(0, 2, 1)
        emb2 = self.embedding_k2(k2).permute(0, 2, 1)
        emb3 = self.embedding_k3(k3).permute(0, 2, 1)
        emb4 = self.embedding_k4(k4).permute(0, 2, 1)
        out1 = self.branch_k1(emb1)
        out2 = self.branch_k2(emb2)
        out3 = self.branch_k3(emb3)
        out4 = self.branch_k4(emb4)
        out1 = out1.view(batch_size, -1)
        out2 = out2.view(batch_size, -1)
        out3 = out3.view(batch_size, -1)
        out4 = out4.view(batch_size, -1)
        combined = torch.cat([out1, out2, out3, out4], dim=1)
        fc_out = F.relu(self.fc(combined))
        output = torch.sigmoid(self.out(fc_out))
        return output

def train_one_epoch(model, train_loader, learning_rate):
    model.train()
    for param_group in model.optimizer.param_groups:
        param_group['lr'] = learning_rate
    epoch_loss_train = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        batch_size = inputs.size(0)
        labels = labels.float().view(-1)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        model.optimizer.zero_grad()
        outputs = model(inputs).view(-1)
        loss = model.criterion(outputs, labels)
        loss.backward()
        model.optimizer.step()
        epoch_loss_train += loss.item() * batch_size
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += batch_size
    epoch_loss_train_avg = epoch_loss_train / total
    epoch_accuracy = correct / total
    return epoch_loss_train_avg, epoch_accuracy

def train_full_dataset():
    print("\n==> Loading full training set")
    data_strong = load_text_file('data_strong_enhancers.txt')
    print("data_strong:", len(data_strong))
    data_weak = load_text_file('data_weak_enhancers.txt')
    print("data_weak:", len(data_weak))
    data_enhancers = data_strong + data_weak
    print("data_enhancers:", len(data_enhancers))
    data_non_enhancers = load_text_file('data_non_enhancers.txt')
    print("data_non_enhancers:", len(data_non_enhancers))
    label_enhancers = np.ones((len(data_enhancers), 1))
    label_non_enhancers = np.zeros((len(data_non_enhancers), 1))
    data = np.concatenate((data_enhancers, data_non_enhancers))
    label = np.concatenate((label_enhancers, label_non_enhancers))
    train_set = EnhancerDataset(data, label)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=4)
    model = EnhancerCnnModel()
    print("CNN Model:", model)
    if torch.cuda.is_available():
        model.cuda()
    train_loss = []
    train_accuracy = []
    for epoch in range(NUMBER_EPOCHS):
        print(f"\n############### EPOCH : {epoch} ###############")
        epoch_loss_train_avg, epoch_accuracy = train_one_epoch(model, train_loader, LEARNING_RATE)
        print(f"Epoch {epoch} - Loss: {epoch_loss_train_avg:.4f} | Accuracy: {epoch_accuracy:.4f}")
        train_loss.append(epoch_loss_train_avg)
        train_accuracy.append(epoch_accuracy)
    file_model = "full_model.pkl"
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, file_model))
    print(f"\nFinal model saved as {file_model} after epoch {NUMBER_EPOCHS-1}")
    with open(os.path.join(MODEL_DIR, f"logfile_loss_full_model.csv"), mode='w', newline='') as lf_loss:
        writer = csv.writer(lf_loss, delimiter=',')
        writer.writerow(['epoch', 'train_loss', 'train_accuracy'])
        for i in range(len(train_loss)):
            writer.writerow([i, train_loss[i], train_accuracy[i]])

if __name__ == "__main__":
    train_full_dataset()
