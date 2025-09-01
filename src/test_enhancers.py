import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import os
import csv
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Hyperparameters
SAMPLE_LENGTH = 200
MY_RANDOM_STATE = 5
torch.manual_seed(MY_RANDOM_STATE)
MODEL_DIR = 'full_training_model'
AVGPOOL1D_KERNEL_SIZE = 4
CONV1D_KERNEL_SIZE = 3
BRANCH_BLOCK1_OUT = 8
BRANCH_BLOCK2_OUT = 16
FULLY_CONNECTED_LAYER_SIZE = 256

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

def read_test_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        m = len(lines) // 5
        my_data = []
        for i in range(m):
            text = lines[i*5+1].strip() + lines[i*5+2].strip() + \
                   lines[i*5+3].strip() + lines[i*5+4].strip()
            my_data.append(text.upper())
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

def calculate_metrics(y_true, y_pred, y_prob):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_prob)
    cm = metrics.confusion_matrix(y_true, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[1, 0]
    fn = cm[0, 1]
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    return {
        'accuracy': accuracy,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcc': mcc,
        'confusion_matrix': cm
    }

def evaluate_model(model_path, test_loader):
    model = EnhancerCnnModel()
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            probs = outputs.cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    return calculate_metrics(all_labels, all_preds, all_probs), all_probs.tolist()

def main():
    print("Loading test data...")
    data_strong = read_test_file('test_strong_enhancer.txt')
    data_weak = read_test_file('test_weak_enhancer.txt')
    data_enhancer = data_strong + data_weak
    data_non_enhancer = read_test_file('test_non_enhancer.txt')
    label_enhancer = np.ones((len(data_enhancer), 1))
    label_non_enhancer = np.zeros((len(data_non_enhancer), 1))
    test_data = np.concatenate((data_enhancer, data_non_enhancer))
    test_labels = np.concatenate((label_enhancer, label_non_enhancer))
    test_dataset = EnhancerDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    model_file = os.path.join(MODEL_DIR, "full_model.pkl")
    results_file = os.path.join(MODEL_DIR, "test_results.csv")
    print(f"Evaluating {os.path.basename(model_file)}...")
    metrics_dict, probs = evaluate_model(model_file, test_loader)
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'MCC'])
        writer.writerow([
            os.path.basename(model_file),
            metrics_dict['accuracy'],
            metrics_dict['auc'],
            metrics_dict['sensitivity'],
            metrics_dict['specificity'],
            metrics_dict['mcc']
        ])
    print("\nTest Results:")
    print(f"Accuracy: {metrics_dict['accuracy']:.4f}")
    print(f"AUC: {metrics_dict['auc']:.4f}")
    print(f"Sensitivity: {metrics_dict['sensitivity']:.4f}")
    print(f"Specificity: {metrics_dict['specificity']:.4f}")
    print(f"MCC: {metrics_dict['mcc']:.4f}")


def plot_training_curves(csv_path):
    data = pd.read_csv(csv_path)

    epochs = data['epoch']
    loss = data['train_loss']
    accuracy = data['train_accuracy']

    plt.figure(figsize=(10, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, marker='o', color='green')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()

# Call this function:
plot_training_curves('logfile_loss_full_model.csv')


if __name__ == "__main__":
    main()