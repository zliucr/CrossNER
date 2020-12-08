import argparse

def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="Cross-domain NER")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="train.log")

    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")

    parser.add_argument("--model_name", type=str, default="bert-base-cased", help="model name (e.g., bert-base-cased, roberta-base)")
    parser.add_argument("--ckpt", type=str, default="", help="reload path for pre-trained model / ner model")
    parser.add_argument("--seed", type=int, default=555, help="random seed (three seeds: 555, 666, 777)")
    parser.add_argument("--tgt_dm", type=str, default="politics", help="target domain")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epoch")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--early_stop", type=int, default=8, help="No improvement after several epoch, we stop training")
    parser.add_argument("--num_tag", type=int, default=0, help="Number of entity in the dataset")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden layer dimension")

    # use BiLSTM
    parser.add_argument("--bilstm", default=False, action="store_true", help="use bilstm-crf structure")
    parser.add_argument("--emb_dim", type=int, default=300, help="embedding dimension")
    parser.add_argument("--n_layer", type=int, default=2, help="number of layers for LSTM")
    parser.add_argument("--emb_file", type=str, default="../glove/glove.6B.300d.txt", help="embeddings file")
    parser.add_argument("--lstm_hidden_dim", type=int, default=256, help="embedding dimension")
    parser.add_argument("--usechar", default=False, action="store_true", help="use character embeddings")
    parser.add_argument("--coach", default=False, action="store_true", help="use coach")
    parser.add_argument("--entity_enc_hidden_dim", type=int, default=300, help="lstm hidden sizes for encoding entity features")
    parser.add_argument("--entity_enc_layers", type=int, default=1, help="lstm encoder layers for encoding entity features")

    # use conll2003 dataset
    parser.add_argument("--conll", default=False, action="store_true", help="use conll2003 ner dataset for pre-training")
    parser.add_argument("--joint", default=False, action="store_true", help="joint train conll2003 data with target domain data")

    # few-shot
    parser.add_argument("--n_samples", type=int, default=-1, help="conduct few-shot learning (10, 25, 40, 55, 70, 85, 100)")

    params = parser.parse_args()

    return params
