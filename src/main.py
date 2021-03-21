import argparse
import json
import spacy
from collections import OrderedDict
from src.dataloader import *
from dataset import *
from src.train import Trainer
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Model parameters")
    parser.add_argument("--folder_params", type=str, default="input", choices=["input"])
    parser.add_argument("--file_params", type=str, default="train", choices=["train"])
    return parser.parse_args(arg_list)


# TODO:
#   save_mode_params, beam_search_params
#    i. gradient clipping (fix train_model- add clip_encoder and clip_decoder) done
#   ii. visualize (tensorboard) (done)
#  iii. print training loss (overfit the data) (done)
#   iv.  create a flow for training (done)
#  create a different optimizer for encoder (if needed) and decoder (done)
#  put self.encoder.unfreeze_model_weights(epoch) outside (in the epochs loop) (done)
#  load best model (done)
#  visualize images (done)


def train_model(trainer, vocab, train_loader, val_loader, optimizer_dec, optimizer_enc, beam_search_params, epochs,
                epoch_start, clip_dec=5.0,  clip_enc=5.0, ** save_model_params):  # enc_starting_epoch
    train_loss_lst = []
    trn_bleu_score_lst = []
    val_bleu_score_lst = []
    apply_analysis = beam_search_params.get('apply_analysis', False)
    for epoch in range(epoch_start, epochs+1):
        # if 0 < enc_starting_epoch <= epoch:
        #     trainer.unfreeze_encoder_weights(epoch)
        #     loss = trainer.train(train_loader, optimizer_dec, optimizer_enc)
        # else:
        #     loss = trainer.train(train_loader, optimizer_dec)
        trainer.unfreeze_encoder_weights(epoch)
        loss = trainer.train(train_loader, optimizer_dec, optimizer_enc, clip_dec, clip_enc)
        train_loss_lst.append(loss)
        # beam search params: k (1, 3), max_seq_len = 35, 'apply_analysis'= True/False
        trn_bleu = trainer.evaluate(train_loader, vocab, beam_search_params)
        val_bleu = trainer.evaluate(val_loader, vocab, beam_search_params, apply_beam_analysis=apply_analysis)

        if save_model_params.get('save_model', False) and val_bleu > val_bleu_score_lst[-1]:
            model_name = save_model_params.get('model_name', 'resnet_lstm')
            if model_name == 'resnet_lstm':
                enc_str = ''
                if optimizer_enc is not None:
                    lr_enc, wd_enc = optimizer_enc.param_group[0]["lr"], optimizer_enc.param_group[0]["weight_decay"]
                    enc_str += f'_enc_params_lr={lr_enc}_wd={wd_enc}'
                lr_dec, wd_dec = optimizer_dec.param_group[0]["lr"], optimizer_dec.param_group[0]["weight_decay"]
                dec_str = f'_dec_params_lr={lr_dec}_wd={wd_dec}'
                save_model_params['model_name'] = f'{model_name}{enc_str}{dec_str}'

            trainer.save_model(epoch, optimizer_dec, optimizer_enc, save_model_params)
            # save_model_params: save_model: True/False, checkpoint_folder, model_name: 'resnet_lstm',
            # checkpoint_folder = 'checkpoint', results_folder = 'results

        trn_bleu_score_lst.append(trn_bleu)
        val_bleu_score_lst.append(val_bleu)
    bleu_score_dict = OrderedDict([('train', trn_bleu_score_lst), ('validation', val_bleu_score_lst)])
    return bleu_score_dict, train_loss_lst


def set_optimizer_params(params, component_name):
    optimizer_dict = {'adam': optim.Adam, 'sgd': optim.SGD}
    rel_params = params["trainer_params"][component_name]['optimizer']
    rel_params['optimizer_func'] = optimizer_dict[rel_params['optimizer_name']]
    return rel_params


def save_bleu_scores(trn_val_score_dict, model_name):
    writer = SummaryWriter()
    names = list(trn_val_score_dict.keys())
    train_bleu = trn_val_score_dict[names[0]]
    val_bleu = trn_val_score_dict[names[1]]
    n_epochs = len(trn_val_score_dict)
    for i in range(n_epochs):  # start_epoch
        tag_scalar_dict = {names[0]: train_bleu[i], names[1]: val_bleu[i]}
        writer.add_scalars(f'bleu/{model_name}', tag_scalar_dict, i + 1)
    writer.close()


def visualize(trainer, data_loader, vocab):
    for (img, caption), references in data_loader:
        img_name = '_'.join(references[0].split(' '))
        trainer.visualize_caption(img, img_name, vocab)


# download: python -m spacy download en (en_core_web_sm) for english spacy tokenization
def main(args):
    root_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    full_path = os.path.join(root_dir, args.folder_params, f'{args.file_params}.json')
    with open(full_path) as json_file:
        run_params = json.load(json_file)
    parser = spacy.load('en_core_web_sm') if run_params["word_preprocessing"] else None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    vocab = Vocabulary(parser)  # freq_thres=2 (default)
    root_input = os.path.join(root_dir, run_params["input_folder"])  # input_folder = 'flickr8k'
    caption_file = os.path.join(root_input, run_params["caption_file"])  # caption_file='captions.txt'
    img_folder = os.path.join(root_input, run_params["img_folder"])  # img_folder = 'images'
    dataset = CaptionDataset(img_folder, caption_file, vocab, transform)
    # ratios=fractions for split (list)
    training_set, validation_set, test_set = train_test_split(dataset, run_params["ratios"])
    # dataloader_params: batch_size: [32, 64], use_collate=True,  n_workers: [8, 2], pin_memory = True
    pad_idx = vocab.stoi['<PAD>']
    train_loader = get_loader(training_set, pad_idx, shuffle=True, batch_size=run_params["batch_size"],
                              ** run_params["dataloader_params"])
    val_loader = get_loader(validation_set, pad_idx, ** run_params["dataloader_params"])
    test_loader = get_loader(test_set, pad_idx, ** run_params["dataloader_params"])

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    vocab_size = len(vocab)
    # cuda_id- the cuda to run
    device = f'cuda:{run_params["cuda_id"]}' if torch.cuda.is_available() else 'cpu'
    epochs = run_params["epochs"]  # change to 120
    enc_dim = run_params["enc_dim"]
    dec_dim = run_params["dec_dim"]
    trainer = Trainer(criterion, device, vocab_size, enc_dim, dec_dim, ** run_params["trainer_params"])
    # optimizer dict: name='encdoer'/'decoder', 'optimizer_name='adam'/'sgd', 'lr'=1e-3/1e-4, 'wd'=0.0/4e-4
    dec_params = set_optimizer_params(run_params, 'decoder')  # a dictionary in dec_params
    optimizer_dec = trainer.set_optimizer('decoder', dec_params)

    unfreeze_params = run_params["trainer_params"]['encoder']['unfreeze_params']
    optimizer_enc = None
    enc_params = None
    if unfreeze_params.get('tune', False):
        enc_params = set_optimizer_params(run_params, 'encoder')
        optimizer_enc = trainer.set_optimizer('encoder', enc_params)
    epoch_start = 1
    if run_params["load_best_model"]:
        # load best model- load_params: model_name, checkpoint_folder
        optimizer_dec, optimizer_enc, epoch_start = trainer.load_model(optimizer_dec, optimizer_enc,
                                                                       **run_params["load_params"])
    print(f'dec_params:\n{dec_params}')
    print(f'enc_params:\n{enc_params}')
    print('successfully build optimizers')
    exit()
    if run_params["train"]:
        # beam search params: k (1, 3), max_seq_len = 35, apply_analysis = True/False
        clip_dec = dec_params.get('clip_val', 5.0)  # 2.0 or even 1.0?
        clip_enc = enc_params.get('clip_val', 5.0) if enc_params is not None else 5.0
        save_model_params = run_params["save_model_params"]
        bleu_score_dict, train_loss_lst = train_model(trainer, vocab, train_loader, val_loader, optimizer_dec,
                                                      optimizer_enc, run_params["beam_search_params"], epochs,
                                                      epoch_start, clip_dec, clip_enc, ** save_model_params)
        if run_params["save_bleu"]:
            model_name = save_model_params.get('model_name', 'resnet_lstm')
            save_bleu_scores(bleu_score_dict, model_name)
        if run_params["print_loss"]:
            for i, loss in enumerate(train_loss_lst, 1):
                print(f'epoch {i}:\ttraining loss = {loss:.4f}')

    if run_params["visualize_caption"]:
        # visualize images of test set via best model, batch size = 1
        visualize(trainer, test_loader, vocab)


if __name__ == "__main__":
    arguments = get_arguments()
    main(arguments)
