import torch
from torch.nn.utils import clip_grad_norm_
import tree as tree_
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext.vocab import FastText, GloVe
import nltk
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from utils import plot_confusion_matrix, plot_classification_report, plot_axes
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from copy import copy
# nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

label_names = ["ROOT_SENTENCE", "SYMBOL", "PUNCT", "AND", "OR", "KEY_C", "KEY_NC", "CONDITION", "VARIABLE",
               "STATEMENT", "CAUSE", "EFFECT", "CAUSE_EFFECT_RELATION", "SEPARATEDCAUSE", "INSERTION",
               "NEGATION", "NONE_CAUSAL", "SEPARATEDSTATEMENT", "SEPARATEDAND", "SENTENCE",
               "SEPARATEDCAUSE_EFFECT_RELATION", "SEPARATEDNEGATION", "WORD", "SEPARATEDNONE_CAUSAL",
               "SEPARATEDOR", "SEPARATEDEFFECT", "SEPARATEDVARIABLE", "SEPARATEDCONDITION"]


class RNN(pl.LightningModule):
    def __init__(self, hparams):
        super(RNN, self).__init__()

        self.outputDim = hparams.outputDim
        self.embeddings = hparams.embeddings
        self.wvecDim = hparams.wvecDim
        self.sample_size = hparams.sample_size
        self.learning_rate = hparams.learning_rate
        self.hparams = hparams
        self.num_words = hparams.num_words

        # Word Vectors
        if self.embeddings.lower() == 'fasttext':
            print("Using FastText Embeddings")
            if self.wvecDim != 300:
                self.wvecDim = 300
                print("Word dimensions set to 300")
            self.L = FastText('en')
        elif self.embeddings.lower() == 'glove':
            print("Using GloVe Embeddings")
            if self.wvecDim != 300:
                self.wvecDim = 300
                print("Word dimensions set to 300")
            self.L = GloVe(name='840B', dim=300)
        elif self.embeddings.lower() == 'bert':
            print("Using BERT Embeddings")
            if self.wvecDim != 768:
                self.wvecDim = 768
                print("Word dimensions set to 300")
            self.wvecDim = 768
        else:
            self.L = nn.Embedding(self.num_words, self.wvecDim)

        # Hidden Activation Weights
        self.W = nn.Linear(2 * self.wvecDim, self.wvecDim, bias=True)

        # Projection Weights/Layer
        self.Ws = nn.Linear(self.wvecDim, self.outputDim, bias=True)

        # Activation Layer: ReLu
        self.activation = F.relu

        # For loss calculation
        self.nodeProbList = []
        self.labelList = []

    # Forward function in PyTorch Lightning is for inference
    def forward(self, sentence, max_beams=10):
        # sentence = sentence
        # max_beams = 10
        # Tokenize Sentence
        # Get Word Embeddings
        if self.embeddings == "BERT":
            sentence = nltk.word_tokenize(sentence)
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True,
                               is_split_into_words=True)
            outputs = model(**inputs)
        else:
            wordMap = tree_.loadWordMap()
            vocab_index, sentence = self.load_vocab_index(sentence.lower(), wordMap)
        wordMapReversed = tree_.loadReversedWordMap()
        # convert tokens into nodes
        nodes = []
        if self.embeddings == "BERT":
            for i in range(1, (len(inputs[0]) - 1)):
                node = tree_.Node(word=inputs[0].tokens[i])
                node.isLeaf = True
                nodes.append(node)
                node.bertEmbedding = outputs[0][0][i]
        elif self.embeddings.lower() == "fasttext":
            for token in sentence:
                node = tree_.Node(word=token)
                node.isLeaf = True
                nodes.append(node)
        else:
            for token in sentence:
                if token in wordMap:
                    node = tree_.Node(word=wordMap[token])
                else:
                    node = tree_.Node(word=wordMap['UNK'])
                node.isLeaf = True
                nodes.append(node)

        initial_nodes = nodes.copy()
        max_steps = len(initial_nodes)
        predictions = [{
            'nodes': initial_nodes.copy(),
            'total_confidence': 0,
            'parent_nodes': []
        }]

        step = 1
        while step < max_steps:
            new_predictions = []
            for prediction in predictions:
                parent_nodes = []
                for node_new in prediction['nodes']:
                    self.forward_rec(node_new, wordMapReversed)
                for i, node_new in enumerate(prediction['nodes']):
                    if i < len(prediction['nodes']) - 1:
                        parent_node = tree_.Node()
                        parent_node.left = copy(node_new)
                        parent_node.right = copy(prediction['nodes'][i + 1])
                        self.forward_rec(parent_node, wordMapReversed)
                        parent_nodes.append({
                            'node': parent_node,
                            'confidence': parent_node.probs[0][torch.argmax(parent_node.probs)].item(),
                            'index': i
                        })
                parent_nodes_sorted = sorted(parent_nodes, key=lambda k: k['confidence'], reverse=True)
                for parent_node_sorted in parent_nodes_sorted:
                    nodes_copy = prediction['nodes'].copy()
                    nodes_copy[parent_node_sorted['index']:(parent_node_sorted['index'] + 2)] = [
                        parent_node_sorted['node']]
                    # print(self.tree_to_string(parent_node_sorted['node'], wordMapReversed).strip())
                    new_prediction = {
                        'nodes': nodes_copy,
                        'total_confidence': prediction['total_confidence'] + parent_node_sorted['confidence'],
                        'parent_nodes': []
                    }
                    prediction_str = "".join(
                        [self.tree_to_string(x, wordMapReversed).strip() for x in new_prediction['nodes']])
                    new_predictions_str = []
                    for y in new_predictions:
                        new_predictions_str.append(
                            "".join([self.tree_to_string(x, wordMapReversed).strip() for x in y['nodes']]))
                    if prediction_str not in new_predictions_str:
                        new_predictions.append(new_prediction)
                    # else:
                    # print("Duplicate found")
            new_predictions = sorted(new_predictions, key=lambda k: k['total_confidence'], reverse=True)
            predictions = new_predictions[:max_beams]
            # for prediction in predictions:
            # print([self.tree_to_string(x, wordMapReversed).strip() for x in prediction['nodes']])
            step = step + 1
        output = []
        for prediction in predictions:
            prediction_string = self.tree_to_string(prediction['nodes'][0], wordMapReversed).strip()
            output.append({
                'prediction': prediction_string,
                'total_confidence': prediction['total_confidence']
            })
        return output

    def forward_rec(self, node, wordMap):
        if node.output is not None:
            return node.output
        if node.isLeaf:
            # get word embedding
            if self.embeddings == 'FastText':
                current_node = self.L[node.word].unsqueeze(0)
            elif self.embeddings == 'BERT':
                current_node = node.bertEmbedding.unsqueeze(0)
            else:
                current_node = self.L(Variable(torch.LongTensor([node.word])))
        else:
            # compute W * [node.left, node.right]
            current_node = self.W(
                torch.cat((self.forward_rec(node.left, wordMap), self.forward_rec(node.right, wordMap)), 1))

        # apply activation layer
        current_node = self.activation(current_node)

        # save probabilities and labels for loss and evaluation
        node.probs = self.Ws(current_node)
        node.label = torch.argmax(node.probs)
        node.output = current_node
        return current_node

    def training_step(self, tree, tree_idx):
        predictions, loss = self.loss_function(tree)
        clip_grad_norm_(model.parameters(), 1.0, norm_type=2.)
        self.log('training/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, tree, tree_idx):
        predictions, loss = self.loss_function(tree)
        n, correct, target, preds = self.evaluate_rec(tree.root)
        self.log('validation/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'val_n': n, 'val_correct': correct, 'target': target, 'preds': preds}

    def validation_epoch_end(
            self,
            outputs):
        val_n = sum([x['val_n'] for x in outputs])
        val_correct = sum([x['val_correct'] for x in outputs])

        if val_n == 0:
            val_accuracy = None
        else:
            val_accuracy = val_correct / val_n

        target = []
        preds = []
        for x in outputs:
            target += x['target']
            preds += x['preds']
        tensorboard = self.logger.experiment
        cm_image = plot_confusion_matrix(target, preds, label_names, tensor_name='validation/confusion_matrix',
                                         normalize=True)
        cr_image = plot_classification_report(target, preds, label_names)
        tensorboard.add_image('validation/confusion_matrix', cm_image, global_step=self.global_step, dataformats='HWC')
        tensorboard.add_image('validation/classification_report', cr_image, global_step=self.global_step,
                              dataformats='HWC')
        self.log('validation/accuracy', val_accuracy, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, tree, tree_idx):
        predictions, loss = self.loss_function(tree)
        n, correct, target, preds = self.evaluate_rec(tree.root)
        self.log('test/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'test_n': n, 'test_correct': correct, 'target': target, 'preds': preds,
                'tree_length': tree.token_length}

    def test_epoch_end(
            self,
            outputs):
        test_n = sum([x['test_n'] for x in outputs])
        test_correct = sum([x['test_correct'] for x in outputs])

        if test_n == 0:
            test_accuracy = None
        else:
            test_accuracy = test_correct / test_n

        df = pd.DataFrame(
            {'Tree Length': [x['tree_length'] for x in outputs],
             'Accuracy': [x['test_correct'] / x['test_n'] for x in outputs]})

        self.log('test/accuracy', test_accuracy, on_step=False, on_epoch=True, prog_bar=False)

        target = []
        preds = []
        for x in outputs:
            target += x['target']
            preds += x['preds']
        tensorboard = self.logger.experiment
        cm_image = plot_confusion_matrix(target, preds, label_names, tensor_name='test/confusion_matrix',
                                         normalize=True)
        cr_image = plot_classification_report(target, preds, label_names)
        print(df.groupby(pd.cut(df["Tree Length"], np.arange(10, 60 + 10, 10))).mean())
        ax_subplot = df.groupby(pd.cut(df["Tree Length"], np.arange(10, 60 + 10, 10))).mean().plot.bar(x='Tree Length')
        ax_subplot.set_xticklabels(['11-20', '21-30', '31-40', '41-50', '51+'])
        figure = ax_subplot.figure
        figure.set_tight_layout(True)
        tensorboard.add_image('test/classification_report', cr_image, dataformats='HWC')
        tensorboard.add_figure('test/accuracy_per_tree_length', figure, close=True)
        tensorboard.add_image('test/confusion_matrix', cm_image, dataformats='HWC')
        tensorboard.add_image('test/confusion_matrix', cm_image, dataformats='HWC')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9,
                                    dampening=0.0)
        return optimizer

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)

    def walk(self, node, wordMap):
        if node.isLeaf:
            # get word embedding
            if self.embeddings.lower() == 'fasttext' or self.embeddings.lower() == 'glove':
                current_node = self.L[node.word].unsqueeze(0)
            elif self.embeddings.lower() == 'bert':
                current_node = node.bertEmbedding.unsqueeze(0)
            else:
                current_node = self.L(Variable(torch.LongTensor([node.word])))
        else:
            # compute W * [node.left, node.right]
            current_node = self.W(
                torch.cat((self.walk(node.left, wordMap), self.walk(node.right, wordMap)), 1))

        if torch.cuda.is_available():
            current_node = current_node.to('cuda')

        # apply activation layer
        current_node = self.activation(current_node)

        # save probabilities and labels for loss and evaluation
        node.probs = self.Ws(current_node)
        self.nodeProbList.append(node.probs)
        self.labelList.append(torch.LongTensor([node.label]))

        return current_node

    def loss_function(self, tree):
        self.nodeProbList = []
        self.labelList = []
        self.walk(tree.root, tree_.loadReversedWordMap())
        self.labelList = Variable(torch.cat(self.labelList))
        nodes = torch.cat(self.nodeProbList)
        predictions = nodes.max(dim=1)[1]
        loss = F.cross_entropy(input=nodes, target=self.labelList)
        return predictions, loss

    def evaluate_rec(self, node):
        n = correct = 0.0

        target = []
        preds = []

        n += 1
        prediction = torch.argmax(node.probs)
        correct += (prediction.item() == node.label)

        target.append(node.label)
        preds.append(prediction.item())

        if node.isLeaf:
            return n, correct, target, preds
        else:
            n_left, correct_left, target_left, preds_left = self.evaluate_rec(node.left)
            n_right, correct_right, target_right, preds_right = self.evaluate_rec(node.right)
            return n + n_left + n_right, correct + correct_left + correct_right, [*target, *target_left,
                                                                                  *target_right], [*preds,
                                                                                                   *preds_left,
                                                                                                   *preds_right]

    def train_dataloader(self):
        trees = tree_.loadTrees(sample_size=args.sample_size, path=self.hparams.tree_path)
        if self.embeddings == 'BERT':
            tree_.initBertEmbeddings(trees)
        dataloader = torch.utils.data.DataLoader(trees, shuffle=True, batch_size=None)
        return dataloader

    def val_dataloader(self):
        trees = tree_.loadTrees(dataSet='dev', sample_size=1.0, path=self.hparams.tree_path)
        if self.embeddings == 'BERT':
            tree_.initBertEmbeddings(trees)
        dataloader = torch.utils.data.DataLoader(trees, batch_size=None, num_workers=0)
        return dataloader

    def test_dataloader(self):
        trees = tree_.loadTrees(dataSet='test', path=self.hparams.tree_path)
        if self.embeddings == 'BERT':
            tree_.initBertEmbeddings(trees)
        dataloader = torch.utils.data.DataLoader(trees, batch_size=None, num_workers=0)
        return dataloader

    def tree_to_string(self, node, wordMap):
        prediction = torch.argmax(node.probs)
        if node.isLeaf:
            if self.embeddings == 'BERT' or self.embeddings.lower() == 'fasttext':
                return " (" + label_names[prediction.item()] + " " + node.word + ")"
            else:
                return " (" + label_names[prediction.item()] + " " + wordMap[node.word] + ")"
        else:
            return " (" + label_names[prediction.item()] + self.tree_to_string(node.left,
                                                                               wordMap) + self.tree_to_string(
                node.right, wordMap) + ")"

    def load_vocab_index(self, input_text, word_map):
        a = []
        tokenized_sentence = nltk.word_tokenize(input_text)
        for token in tokenized_sentence:
            # we only saved lower case words
            if token.lower() in word_map:
                a.append(word_map[token.lower()])
            else:
                print("Unkown token in the sentence " + token)
                a.append(word_map['UNK'])
        return a, tokenized_sentence


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('-p', '--path', dest='model_path', required=False)
    parser.add_argument('-s', '--sample_size', dest='sample_size', default=1.0, type=float, required=False)
    parser.add_argument('-d', '--dim', dest='wvecDim', type=int, default=300, required=False)
    parser.add_argument('-em', '--embeddings', dest='embeddings', default='random', required=False)
    parser.add_argument('-o', '--output_dimensions', dest='outputDim', default=28, type=int, required=False)
    parser.add_argument('-tp' '--tree_path', dest='tree_path', default='/trees/final_trees/Experiment2_without_sub',
                        required=False)
    parser.add_argument('-lr' '--learning_rate', dest='learning_rate', default=0.01, type=float,
                        required=False)

    args = parser.parse_args()

    # For debugging purposes only
    # torch.autograd.set_detect_anomaly(True)
    # args.fast_dev_run = True

    # Get Dataset Type
    args.dataset_type = 'unknown'
    if args.tree_path == '/trees/final_trees/Experiment2_without_sub':
        args.dataset_type = 'left_branching'
    elif args.tree_path == '/trees/final_trees/Experiment3_right_branching':
        args.dataset_type = 'right_branching'

    tree_.buildWordMap()
    args.num_words = len(tree_.loadWordMap())

    early_stop_callback = EarlyStopping(
        monitor='validation/accuracy',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='max',
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='validation/loss_epoch',
        save_top_k=3,
        filename='epoch{epoch:02d}'
    )

    #model = RNN(hparams=args)
    model = RNN.load_from_checkpoint(args.model_path, hparams=args)
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/', name=(
            'embeddings=%s,dim=%d,dataset=%s,max_epochs=%s' % (
        args.embeddings, args.wvecDim, args.dataset_type, args.max_epochs)))
    args.logger = tb_logger
    args.callbacks = [early_stop_callback, checkpoint_callback]
    trainer = pl.Trainer.from_argparse_args(args)
    #trainer.fit(model)
    #print(checkpoint_callback.best_model_path)
    trainer.test(model)
