from rnn import RNN
from argparse import ArgumentParser
from rnn_temperature_scaling import RNNWithTemperature
from flask import Flask, current_app
from flask_cors import CORS
from flask import request
from argparse import Namespace

args = dict(
    PATH='lightning_logs/embeddings=BERT,dim=768,dataset=left_branching,max_epochs=50/version_1/checkpoints/epochepoch=14.ckpt',
    PATH_TEMPERATURE='lightning_logs/embeddings=BERT,dim=768,dataset=left_branching,max_epochs=None/temperature_scaling/version_8/checkpoints/epoch=16-step=2315.ckpt',
    PATH_BERT_RIGHT='lightning_logs/embeddings=BERT,dim=768,dataset=right_branching,max_epochs=50/version_0/checkpoints/epochepoch=05.ckpt',
    PATH_FASTTEXT_LEFT='lightning_logs/embeddings=FastText,dim=300,dataset=left_branching,max_epochs=50/version_0/checkpoints/epochepoch=10.ckpt',
    PATH_FASTTEXT_RIGHT='lightning_logs/embeddings=FastText,dim=300,dataset=right_branching,max_epochs=50/version_0/checkpoints/epochepoch=09.ckpt',
    PATH_RANDOM_LEFT='lightning_logs/embeddings=random,dim=300,dataset=left_branching,max_epochs=50/version_0/checkpoints/epochepoch=11.ckpt',
    PATH_RANDOM_RIGHT='lightning_logs/embeddings=random,dim=300,dataset=right_branching,max_epochs=50/version_0/checkpoints/epochepoch=05.ckpt',
    sample_size=1.0,
    wvecDim=768,
    embeddings='BERT',
    outputDim=28,
    tree_path='/trees/final_trees/Experiment2_without_sub',
    learning_rate=0.01,
    num_words=4365
)
args = Namespace(**args)

# BERT
model_bert = RNN.load_from_checkpoint(args.PATH, hparams=args)
model_bert.eval()
model_bert_ts = None

model_bert_right = RNN.load_from_checkpoint(args.PATH_BERT_RIGHT, hparams=args)
model_bert_right.eval()

if args.PATH_TEMPERATURE is not None:
    model_bert_ts = RNNWithTemperature.load_from_checkpoint(args.PATH_TEMPERATURE, model=model_bert)
    model_bert_ts.eval()

# FastText
args.wvecDim = 300
args.embeddings = "FastText"

model_fasttext_left = RNN.load_from_checkpoint(args.PATH_FASTTEXT_LEFT, hparams=args)
model_fasttext_left.eval()

model_fasttext_right = RNN.load_from_checkpoint(args.PATH_FASTTEXT_RIGHT, hparams=args)
model_fasttext_right.eval()

# Random
args.embeddings = "random"

model_random_left = RNN.load_from_checkpoint(args.PATH_RANDOM_LEFT, hparams=args)
model_random_left.eval()

model_random_right = RNN.load_from_checkpoint(args.PATH_RANDOM_RIGHT, hparams=args)
model_random_right.eval()

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return current_app.send_static_file('index.html')

@app.route('/index.css')
def css():
    return current_app.send_static_file('index.css')

@app.route('/favicon.ico')
def favicon():
    return current_app.send_static_file('favicon.ico')

@app.route('/predict', methods=['POST'])
def predict():
    request.get_json(force=True)
    if int(request.json['max_beams']) < 1:
        request.json['max_beams'] = '1'

    if request.json["embeddings"] == 'BERT':
        if request.json["dataset"] == 'left_branching':
            if request.json["temperature_scaling"]:
                print('temp scaling')
                model = model_bert_ts
            else:
                print('no temp scaling')
                model = model_bert
        else:
            model = model_bert_right
    elif request.json["embeddings"] == 'FastText':
        print('FastText')
        if request.json["dataset"] == 'left_branching':
            model = model_fasttext_left
        else:
            model = model_fasttext_right
    else:
        print('Random')
        # model = model_fasttext_left
        if request.json["dataset"] == 'left_branching':
            model = model_random_left
        else:
            model = model_random_right
    return model(request.json['sentence'], max_beams=int(request.json['max_beams']))[0]['prediction']

