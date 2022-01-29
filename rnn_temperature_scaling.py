import torch
from torch.nn.utils import clip_grad_norm_
import tree as tree_
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import nltk
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from rnn import RNN
from copy import deepcopy, copy

# nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

label_names = ["ROOT_SENTENCE", "SYMBOL", "PUNCT", "AND", "OR", "KEY_C", "KEY_NC", "CONDITION", "VARIABLE",
               "STATEMENT", "CAUSE", "EFFECT", "CAUSE_EFFECT_RELATION", "SEPARATEDCAUSE", "INSERTION",
               "NEGATION", "NONE_CAUSAL", "SEPARATEDSTATEMENT", "SEPARATEDAND", "SENTENCE",
               "SEPARATEDCAUSE_EFFECT_RELATION", "SEPARATEDNEGATION", "WORD", "SEPARATEDNONE_CAUSAL",
               "SEPARATEDOR", "SEPARATEDEFFECT", "SEPARATEDVARIABLE", "SEPARATEDCONDITION"]

class RNNWithTemperature(pl.LightningModule):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model):
        super(RNNWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, sentence, max_beams=10):
        # sentence = sentence
        # max_beams = 10
        # Tokenize Sentence
        # Get Word Embeddings
        if self.model.embeddings == "BERT":
            sentence = nltk.word_tokenize(sentence)
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True,
                               is_split_into_words=True)
            outputs = model(**inputs)
        else:
            wordMap = tree_.loadWordMap()
            vocab_index, sentence = self.model.load_vocab_index(sentence.lower(), wordMap)
        wordMapReversed = tree_.loadReversedWordMap()
        # convert tokens into nodes
        nodes = []
        if self.model.embeddings == "BERT":
            for i in range(1, (len(inputs[0]) - 1)):
                node = tree_.Node(word=inputs[0].tokens[i])
                node.isLeaf = True
                nodes.append(node)
                node.bertEmbedding = outputs[0][0][i]
        else:
            for token in sentence:
                node = tree_.Node(word=wordMap[token])
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
                    nodes_copy[parent_node_sorted['index']:(parent_node_sorted['index'] + 2)] = [parent_node_sorted['node']]
                    # print(self.model.tree_to_string(parent_node_sorted['node'], wordMapReversed).strip())
                    new_prediction = {
                        'nodes': nodes_copy,
                        'total_confidence': prediction['total_confidence'] + parent_node_sorted['confidence'],
                        'parent_nodes': []
                    }
                    prediction_str = "".join([self.model.tree_to_string(x, wordMapReversed).strip() for x in new_prediction['nodes']])
                    new_predictions_str = []
                    for y in new_predictions:
                        new_predictions_str.append("".join([self.model.tree_to_string(x, wordMapReversed).strip() for x in y['nodes']]))
                    if prediction_str not in new_predictions_str:
                        new_predictions.append(new_prediction)
                    #else:
                        #print("Duplicate found")
            new_predictions = sorted(new_predictions, key=lambda k: k['total_confidence'], reverse=True)
            predictions = new_predictions[:max_beams]
            #for prediction in predictions:
                #print([self.model.tree_to_string(x, wordMapReversed).strip() for x in prediction['nodes']])
            step = step + 1
        output = []
        for prediction in predictions:
            prediction_string = self.model.tree_to_string(prediction['nodes'][0], wordMapReversed).strip()
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
            if self.model.embeddings == 'FastText':
                current_node = self.model.L[wordMap[node.word]].unsqueeze(0)
            elif self.model.embeddings == 'BERT':
                current_node = node.bertEmbedding.unsqueeze(0)
            else:
                current_node = self.model.L(Variable(torch.LongTensor([node.word])))
        else:
            # compute W * [node.left, node.right]
            current_node = self.model.W(
                torch.cat((self.forward_rec(node.left, wordMap), self.forward_rec(node.right, wordMap)), 1))

        # apply activation layer
        current_node = self.model.activation(current_node)

        # save probabilities and labels for loss and evaluation
        node.probs = nn.Softmax(dim=1)(self.temperature_scale(self.model.Ws(current_node)))
        node.label = torch.argmax(node.probs)
        node.output = current_node
        return current_node

    def training_step(self, tree, tree_idx):
        predictions, loss = self.loss_function(tree)
        clip_grad_norm_(self.parameters(), 1.0, norm_type=2.)
        self.log('training/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def train_dataloader(self):
        trees = tree_.loadTrees(dataSet='dev', sample_size=1.0, path=self.model.hparams.tree_path)
        if self.model.embeddings == 'BERT':
            tree_.initBertEmbeddings(trees)
        dataloader = torch.utils.data.DataLoader(trees, batch_size=None, num_workers=0)
        return dataloader

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def loss_function(self, tree):
        self.model.nodeProbList = []
        self.model.labelList = []
        self.model.walk(tree.root, tree_.loadReversedWordMap())
        self.model.labelList = Variable(torch.cat(self.model.labelList))
        logits = torch.cat(self.model.nodeProbList)
        predictions = logits.max(dim=1)[1]
        loss = F.cross_entropy(input=self.temperature_scale(logits), target=self.model.labelList)
        return predictions, loss

    def configure_optimizers(self):
        # optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        optimizer = torch.optim.SGD([self.temperature], lr=0.01, momentum=0.9,
                                    dampening=0.0)
        return optimizer

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('-p', '--path', dest='PATH', required=True)
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

    args.num_words = len(tree_.loadWordMap())

    model = RNN.load_from_checkpoint(args.PATH, hparams=args)
    model.eval()
    model_with_temperature_scaling = RNNWithTemperature(model)
    print("Model structure: ", model_with_temperature_scaling, "\n")
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/', name=(
            'embeddings=%s,dim=%d,dataset=%s,max_epochs=%s/temperature_scaling' % (
        args.embeddings, args.wvecDim, args.dataset_type, args.max_epochs)))
    args.logger = tb_logger
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model_with_temperature_scaling)