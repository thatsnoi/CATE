import collections
import re
import utils
from random import sample
from transformers import AutoTokenizer, AutoModel
import os
import torch
import time
import gc
from copy import copy
log = utils.get_logger()
UNK = 'UNK'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


class Node:
    def __init__(self, label=None, word=None):
        self.label = label
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False
        self.fprop = False
        self.probs = None
        self.bertEmbedding = None
        self.output = None

    def __copy__(self):
        node = Node()
        node.label = self.label
        node.word = self.word
        if self.parent is not None:
            node.parent = copy(self.parent)
        if self.left is not None:
            node.left = copy(self.left)
        if self.right is not None:
            node.right = copy(self.right)
        node.isLeaf = self.isLeaf
        node.fprop = self.fprop
        node.probs = self.probs
        node.bertEmbedding = self.bertEmbedding
        return node

    #def __str__(self):
    #    return str(self.label)


class Tree:

    def __init__(self, treeString, openChar='(', closeChar=')', line_in_dataset=None, path=None):
        tokens = []
        self.open = '('
        self.close = ')'
        tokens = re.findall(r'\(|\)|[^\(\) ]+', treeString.rstrip("\n"))
        # for toks in treeString.strip().split():
        #     tokens += list(toks)
        self.tokens = tokens
        self.token_length = 0
        self.line_in_dataset = line_in_dataset
        self.path = path
        self.root = self.parse(tokens)

    def __len__(self):
        return 1 # hot fix PyTorch Lightning

    def parse(self, tokens, parent=None):
        if tokens[0] != self.open:
            print("Malformed Tree in line " + str(self.line_in_dataset))
        if tokens[-1] != self.close:
            print("Malformed Tree in line " + str(self.line_in_dataset))
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2  # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open and tokens[split + 1] != self.close:
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if (split + 1) >= len(tokens): print(tokens)
            if tokens[split] == self.open and tokens[split + 1] != self.close:
                countOpen += 1
            if tokens[split] == self.close:
                if tokens[split - 1] == "2" and tokens[split - 2] != "38":
                    pass
                else:
                    countClose += 1
            split += 1

        # New node
        node = Node(int(tokens[1]) - 1)  # zero index labels
        node.parent = parent

        # leaf Node
        if countOpen == 0:
            # print(tokens[2:-1])
            # print(''.join(tokens[2:-1]).lower())
            node.word = ''.join(tokens[2:-1]).lower()  # lower case?
            # print(node.word)
            node.isLeaf = True
            self.token_length += 1
            return node

        node.left = self.parse(tokens[2:split], parent=node)
        node.right = self.parse(tokens[split:-1], parent=node)
        return node


def leftTraverse(root, nodeFn=None, args=None):
    """
    Recursive function traverses tree
    from left to right.
    Calls nodeFn at each node
    """
    nodeFn(root, args)
    if root.left is not None:
        leftTraverse(root.left, nodeFn, args)
    if root.right is not None:
        leftTraverse(root.right, nodeFn, args)


def countWords(node, words):
    if node.isLeaf:
        words[node.word] += 1


def mapWords(node, wordMap):
    if node.isLeaf:
        if node.word is not None and node.word not in wordMap:
            node.word = wordMap[UNK]
        else:
            node.word = wordMap[node.word]


def loadWordMap():
    import pickle
    with open('wordMap.bin', 'rb') as fid:
        wordMap = pickle.load(fid, encoding='bytes')
        return {key: val for key, val in wordMap.items()}


def loadReversedWordMap():
    import pickle
    with open('wordMapReversed.bin', 'rb') as fid:
        wordMap = pickle.load(fid, encoding='bytes')
        wordMap = {key: val for key, val in wordMap.items()}
        wordMap[4364] = UNK
        return wordMap


def buildWordMap(path='trees/final_trees/Experiment2_without_sub'):
    """
    Builds map of all words in training set
    to integer values.
    """
    import pickle
    file = path + '/train.txt'
    log.info("Reading trees..")
    trees = []
    i = 1
    with open(file, 'r') as fid:
        for l in fid.readlines():
            tree = Tree(l, line_in_dataset=i, path=file)
            trees.append(tree)
            i += 1

    log.info("Counting words..")
    words = collections.defaultdict(int)
    reversedWords = collections.defaultdict(int)
    for tree in trees:
        leftTraverse(tree.root, nodeFn=countWords, args=words)

    wordMap = dict(zip(words.__iter__(), range(len(words))))
    wordMapReversed = dict(zip(range(len(words)), words.__iter__()))
    wordMap[UNK] = len(words)  # Add unknown as word

    print("This is the word map")
    print(wordMap)

    print("This is the reversed word map")
    print(wordMapReversed)

    with open('wordMap.bin', 'wb') as fid:
        pickle.dump(wordMap, fid)

    with open('wordMapReversed.bin', 'wb') as fid:
        pickle.dump(wordMapReversed, fid)


def loadTrees(dataSet='train', sample_size=None, path='/trees/final_trees/Experiment3_right_branching'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    wordMap = loadWordMap()
    # file = os.getcwd() + '/trees/%s.txt' % dataSet
    file = os.getcwd() + path + '/%s.txt' % dataSet
    log.info("Reading trees..")
    i = 1
    trees = []
    with open(file, 'r') as fid:
        for l in fid.readlines():
            tree = Tree(l, line_in_dataset=i, path=file)
            trees.append(tree)
            i += 1

    if sample_size is not None:
        assert isinstance(sample_size, int) or isinstance(sample_size, float), \
            "The sample size has to be either int (number of elements) or float (fraction of the initial dataset)"
        if isinstance(sample_size, int):
            assert 0 < sample_size <= len(trees), \
                "Wrong sample size; if it's an integer then must be in [1," + len(trees) + "]"
        if isinstance(sample_size, float):
            assert 0 < sample_size <= 1, \
                "Wrong sample size; if it's a float must be in (0, 1]"
            sample_size = int(sample_size * len(trees))
            assert sample_size > 0, "Sample fraction too small"
        if sample_size < len(trees):
            trees = sample(trees, sample_size)

    for tree in trees:
        leftTraverse(tree.root, nodeFn=mapWords, args=wordMap)
    return trees


def flattenSentences(trees):
    wordMap = loadReversedWordMap()
    sentences = []
    for i, tree in enumerate(trees):
        tokens = []
        getTokens(tree.root, wordMap, tokens)
        sentences.append(tokens)
    return sentences


def initBertEmbeddings(trees):
    wordMap = loadReversedWordMap()
    batched_tokens = []
    for i, tree in enumerate(trees):
        tokens = []
        getTokens(tree.root, wordMap, tokens)
        batched_tokens.append(tokens)

    print("Tokenizing...")
    for i, tree in enumerate(trees):
        inputs = tokenizer(batched_tokens[i], return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)
        if i % 10 == 9:
            print("%d/%d" % ((i + 1), len(trees)))
        outputs = model(**inputs)
        alignEmbeddings(tree.root, wordMap, inputs[0], outputs[0][0], 1)
        gc.collect()
    time.sleep(10)
    #print("Aligning Embeddings...")
    #for i, tree in enumerate(trees):
        #alignEmbeddings(tree.root, wordMap, inputs[i], outputs[0][i], 1)

    print("Initialized Bert Embeddings")


# https://www.geeksforgeeks.org/print-leaf-nodes-left-right-binary-tree/
def getTokens(node, wordMap, tokens) -> None:
    # If node is null, return
    if not node:
        return

    # If node is leaf node,
    # print its data
    if (not node.left and
            not node.right):
        tokens.append(wordMap[node.word])
        return

    # If left child exists,
    # check for leaf recursively
    if node.left:
        getTokens(node.left, wordMap, tokens)

    # If right child exists,
    # check for leaf recursively
    if node.right:
        getTokens(node.right, wordMap, tokens)


# TODO Split nodes if multiple tokens per word
def alignEmbeddings(node, wordMap, bertTokens, bertOutput, index):
    # If node is null, return
    if not node:
        return

    # If node is leaf node,
    # print its data
    if (not node.left and
            not node.right):
        node.bertEmbedding = bertOutput[index].to(device)
        # tokens.append(wordMap[node.word])
        current_node = node
        while bertTokens.offsets[index + 1][0] != 0:
            index += 1
            current_node.right = Node(label=current_node.label, word=bertTokens.tokens[index])
            current_node = current_node.right
            current_node.bertEmbedding = bertOutput[index].to(device)
        return index + 1

    # If left child exists,
    # check for leaf recursively
    if node.left:
        index = alignEmbeddings(node.left, wordMap, bertTokens, bertOutput, index)

    # If right child exists,
    # check for leaf recursively
    if node.right:
        index = alignEmbeddings(node.right, wordMap, bertTokens, bertOutput, index)

    return index


if __name__ == '__main__':
    buildWordMap()
    train = loadTrees()
