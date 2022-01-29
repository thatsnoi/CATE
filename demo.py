from rnn import RNN
from argparse import ArgumentParser
from rnn_temperature_scaling import RNNWithTemperature

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', dest='PATH', required=True)
    parser.add_argument('-pt', '--path_temp', dest='PATH_TEMPERATURE', required=False, default=None)

    parser.add_argument('-s', '--sample_size', dest='sample_size', default=1.0, type=float, required=False)
    parser.add_argument('-d', '--dim', dest='wvecDim', type=int, default=300, required=False)
    parser.add_argument('-em', '--embeddings', dest='embeddings', default='random', required=False)
    parser.add_argument('-o', '--output_dimensions', dest='outputDim', default=28, type=int, required=False)
    parser.add_argument('-tp' '--tree_path', dest='tree_path', default='/trees/final_trees/Experiment2_without_sub',
                        required=False)
    parser.add_argument('-lr' '--learning_rate', dest='learning_rate', default=0.01, type=float,
                        required=False)
    args = parser.parse_args()
    args.num_words = 4365
    default_model = RNN.load_from_checkpoint(args.PATH, hparams=args)
    default_model.eval()
    model = None
    if args.PATH_TEMPERATURE is not None:
        model = RNNWithTemperature.load_from_checkpoint(args.PATH_TEMPERATURE, model=default_model)
        model.eval()

    input_sample = "If set to true, the widget is visible."
    result = model(input_sample)

    print(result)
