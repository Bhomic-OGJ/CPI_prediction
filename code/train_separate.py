import pickle
import sys
import timeit

import numpy as np

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self, n_fingerprint, n_word, dim, layer_gnn, layer_rnn, layer_output):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        # GRU replaces CNN
        self.rnn = nn.GRU(input_size=dim, hidden_size=dim, num_layers=layer_rnn, batch_first=True)
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_rnn(self, x, xs):
        # xs: (L, dim) -> make (1, L, dim)
        seq = torch.unsqueeze(xs, 0)
        outputs, _ = self.rnn(seq)  # (1, L, dim)
        hs = torch.relu(self.W_attention(outputs.squeeze(0)))  # (L, dim)

        h = torch.relu(self.W_attention(x))  # (1, dim)
        weights = torch.tanh(F.linear(h, hs))  # (1, L)
        ys = torch.t(weights) * hs  # (L, dim)

        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):
        fingerprints, adjacency, words = inputs
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, len(self.W_gnn))

        word_vectors = self.embed_word(words)
        protein_vector = self.attention_rnn(compound_vector, word_vectors)

        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(len(self.W_out)):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        return interaction

    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, predicted_labels, predicted_scores) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        accuracy = accuracy_score(T, Y)
        return AUC, precision, recall, accuracy

    def save_AUCs(self, metrics, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, metrics)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype, device):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def stratified_split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1234):
    """Split dataset into train/val/test with stratification on protein sequences.

    This implementation performs the stratified splits using integer indices
    (np.arange) so we don't coerce torch.Tensor objects into NumPy arrays and
    therefore avoid the FutureWarning.
    """
    protein_sequences = [item[2] for item in dataset]
    unique_proteins = {}
    protein_labels = []
    for seq in protein_sequences:
        if isinstance(seq, (list, np.ndarray)):
            seq_str = str(tuple(seq))
        elif isinstance(seq, str):
            seq_str = seq
        else:
            seq_str = str(seq)
        if seq_str not in unique_proteins:
            unique_proteins[seq_str] = len(unique_proteins)
        protein_labels.append(unique_proteins[seq_str])

    indices = np.arange(len(dataset))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, val_test_idx = next(sss.split(indices, protein_labels))

    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    sss_val_test = StratifiedShuffleSplit(n_splits=1, train_size=val_test_ratio, random_state=seed)

    val_test_positions = val_test_idx
    val_test_labels = [protein_labels[i] for i in val_test_positions]
    val_pos, test_pos = next(sss_val_test.split(val_test_positions, val_test_labels))

    val_idx = val_test_positions[val_pos]
    test_idx = val_test_positions[test_pos]

    dataset_train = [dataset[i] for i in train_idx]
    dataset_val = [dataset[i] for i in val_idx]
    dataset_test = [dataset[i] for i in test_idx]

    print(f"Train set size: {len(dataset_train)} ({len(dataset_train)/len(dataset):.2%})")
    print(f"Validation set size: {len(dataset_val)} ({len(dataset_val)/len(dataset):.2%})")
    print(f"Test set size: {len(dataset_test)} ({len(dataset_test)/len(dataset):.2%})")
    return dataset_train, dataset_val, dataset_test


if __name__ == "__main__":
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_rnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, layer_gnn, window, layer_rnn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_rnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    dir_input = ('../dataset/' + DATASET + '/input/'
                 'radius' + radius + '_ngram' + ngram + '/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor, device)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor, device)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor, device)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor, device)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset, 1234)

    dataset_train, dataset_dev, dataset_test = stratified_split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1234)

    torch.manual_seed(1234)
    model = CompoundProteinInteractionPrediction(n_fingerprint, n_word, dim, layer_gnn, layer_rnn, layer_output).to(device)
    trainer = Trainer(model, lr, weight_decay)
    tester = Tester(model)

    # Create GRU-specific output directories so existing outputs remain intact
    base_out = '../output/gru'
    result_out = os.path.join(base_out, 'result')
    model_out_root = os.path.join(base_out, 'model')
    os.makedirs(result_out, exist_ok=True)
    os.makedirs(model_out_root, exist_ok=True)

    file_AUCs = os.path.join(result_out, 'AUCs--' + setting + '.txt')
    # Per-setting model directory to avoid overwriting other settings
    model_dir = os.path.join(model_out_root, setting)
    os.makedirs(model_dir, exist_ok=True)
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
            'AUC_test\tPrecision_test\tRecall_test\tAccuracy_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    for epoch in range(1, iteration):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        AUC_dev = tester.test(dataset_dev)[0]
        AUC_test, precision_test, recall_test, accuracy = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev,
                AUC_test, precision_test, recall_test, accuracy]
        tester.save_AUCs(AUCs, file_AUCs)
        # Save per-epoch model in the GRU-specific model directory
        model_filename = os.path.join(model_dir, f'model_epoch{epoch}.pt')
        tester.save_model(model, model_filename)

        print('\t'.join(map(str, AUCs)))
