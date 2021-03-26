import argparse
import spacy
from collections import OrderedDict
from dataloader import *
from dataset import *
from train import Trainer
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
#   check correctness of unfreeze (change epoch in unfreeze params to 3 instead of 13)
#   check save and load model
#   set model_name outside loop (including teacher forcing ratio (tfr) and (lstm) dropout (done)
#   problems in optimizer_enc (check correctness) done
#   beam_search_params  (done)
#    i. gradient clipping (fix train_model- add clip_encoder and clip_decoder) done
#   ii. visualize (tensorboard) (done)
#  iii. print training loss (overfit the data) (done)
#   iv.  create a flow for training (done)
#  create a different optimizer for encoder (if needed) and decoder (done)
#  put self.encoder.unfreeze_model_weights(epoch) outside (in the epochs loop) (done)
#  load best model (done)
#  visualize images (done)


def train_model(trainer, vocab, train_loader, val_loader, optimizer_dec, params_enc, beam_search_params, epochs,
                epoch_start, clip_dec=5.0,  clip_enc=5.0, verbose=False, ** save_model_params):
    train_loss_lst = []
    trn_bleu_score_lst = []
    val_bleu_score_lst = []
    apply_analysis = beam_search_params.get('apply_analysis', False)
    optimizer_enc = None
    for epoch in range(epoch_start, epochs+1):
        print(f'epoch: {epoch}')
        trainer.unfreeze_encoder_weights(epoch)
        if trainer.has_unfroze_encoder:
            optimizer_enc = trainer.set_optimizer('encoder', params_enc) if optimizer_enc is None else optimizer_enc
        loss = trainer.train(train_loader, optimizer_dec, optimizer_enc, clip_dec, clip_enc)
        train_loss_lst.append(loss)
        # trn_bleu = trainer.evaluate(train_loader, vocab, beam_search_params)
        val_bleu = trainer.evaluate(val_loader, vocab, beam_search_params, apply_beam_analysis=apply_analysis)

        if verbose:
            print(f'training loss = {loss:.4f}')
            # print(f'training bleu score = {trn_bleu:.4f}\tvalidation bleu score = {val_bleu:.4f}')
            print(f'validation bleu score = {val_bleu:.4f}')

        if save_model_params.get('save_model', False) and val_bleu > val_bleu_score_lst[-1]:
            trainer.save_model(epoch, optimizer_dec, optimizer_enc, save_model_params)
            # save_model_params: save_model: True/False, checkpoint_folder, model_name: 'resnet_lstm',
            # checkpoint_folder = 'checkpoint', results_folder = 'results

        # trn_bleu_score_lst.append(trn_bleu)
        val_bleu_score_lst.append(val_bleu)
    bleu_score_dict = OrderedDict([('train', trn_bleu_score_lst), ('validation', val_bleu_score_lst)])
    return bleu_score_dict, train_loss_lst


def set_component_name(component_dict, params_lst, has_model_name=True):
    params_str = ''
    for param in params_lst:
        if param == 'lr' or param == 'wd':
            params_str += f'_{param}={str(component_dict["optimizer"][param]).replace(".","")}'
        else:
            params_str += f'_{param}={str(component_dict[param]).replace(".","")}'
    component_name = ''
    if has_model_name:
        component_name = component_dict["model_name"]
    component_str = f'{component_name}{params_str}'
    return component_str


def set_model_name(trainer_params):
    enc_str = set_component_name(trainer_params['encoder'], ["lr", "wd"])
    dec_str = set_component_name(trainer_params['decoder'], ["lr", "wd", "dropout"])
    seq_str = set_component_name(trainer_params['seq2seq'], ["teacher_force_ratio"], False)
    model_name = f'{enc_str}_{dec_str}_{seq_str}'
    return model_name


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
    # root_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    root_dir = os.getcwd()
    full_path = os.path.join(root_dir, args.folder_params, f'{args.file_params}.json')
    with open(full_path) as json_file:
        run_params = json.load(json_file)
    parser = spacy.load('en_core_web_sm')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    vocab = Vocabulary(parser)  # freq_thres=2 (default)

    root_input = os.path.join(root_dir, run_params["input_folder"])  # input_folder = 'flickr8k'
    load_folder = os.path.join(root_dir, run_params["load_folder"])  # load_folder='preprocess'
    img_folder = os.path.join(root_input, run_params["img_folder"])  # img_folder = 'images'
    dataset = CaptionDataset(img_folder, load_folder, vocab, transform)
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
    params_dec = set_optimizer_params(run_params, 'decoder')  # a dictionary in params_dec
    optimizer_dec = trainer.set_optimizer('decoder', params_dec)

    unfreeze_params = run_params["trainer_params"]['encoder']['unfreeze_params']
    params_enc = set_optimizer_params(run_params, 'encoder')
    # if unfreeze_params.get('tune', False):
    #     params_enc = set_optimizer_params(run_params, 'encoder')
    epoch_start = 1
    if run_params["load_best_model"]:
        # load optimizer_enc
        optimizer_enc = None
        if unfreeze_params.get('tune', False):
            epoch_freeze = run_params["load_params"]["epoch"]
            params_enc = set_optimizer_params(run_params, 'encoder')
            trainer.unfreeze_encoder_weights(epoch_freeze)
            optimizer_enc = trainer.set_optimizer('encoder', params_enc)
        # load best model- load_params: model_name, checkpoint_folder
        optimizer_dec, optimizer_enc, epoch_start = trainer.load_model(optimizer_dec, optimizer_enc,
                                                                       **run_params["load_params"])
        beam_search_params = run_params["beam_search_params"]
        test_bleu_score = trainer.evaluate(test_loader, vocab, beam_search_params)
        print(f'test bleu score = {test_bleu_score}')
    if run_params["train"]:
        # beam search params: k (1, 3), max_seq_len = 35, apply_analysis = True/False
        clip_dec = params_dec.get('clip_val', 5.0)  # 2.0 or even 1.0?
        clip_enc = params_enc.get('clip_val', 5.0) if params_enc is not None else 5.0
        verbose = run_params.get('verbose', False)
        save_model_params = run_params["save_model_params"]
        model_name = set_model_name(run_params["trainer_params"])
        save_model_params["model_name"] = model_name
        beam_search_params = run_params["beam_search_params"]
        # beam search params: k (1, 3), max_seq_len = 35, 'apply_analysis'= True/False
        bleu_score_dict, train_loss_lst = train_model(trainer, vocab, train_loader, val_loader, optimizer_dec,
                                                      params_enc, beam_search_params, epochs, epoch_start,
                                                      clip_dec, clip_enc, verbose, ** save_model_params)
        if save_model_params.get('save_model', False):
            test_bleu_score = trainer.evaluate(test_loader, vocab, beam_search_params)
            print(f'test bleu score = {test_bleu_score}')
        if run_params["save_bleu"]:
            model_name = save_model_params.get('model_name', 'resnet50_lstm')
            save_bleu_scores(bleu_score_dict, model_name)

    if run_params["visualize_caption"]:
        # visualize images of test set via best model, batch size = 1
        visualize(trainer, test_loader, vocab)


if __name__ == "__main__":
    arguments = get_arguments()
    main(arguments)
