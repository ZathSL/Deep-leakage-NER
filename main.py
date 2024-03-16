import argparse
import copy
import math
import os
import pickle
import time

from torch.utils.data import DataLoader
from sklearn import metrics
import dlg_attack
import util
import torch
from datasets import ClassLabel, load_dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForTokenClassification, PretrainedConfig, default_data_collator,
    DataCollatorForTokenClassification,
)


def main(args):
    # Disabilita l'utilizzo della libreria cuDNN (Cuda Deep Neural Network library)
    # durante l'esecuzione delle operazioni di deep learning su GPU
    #torch.backends.cudnn.enabled = False

    # Scelta del dispositivo su cui eseguire le operazioni di deep learning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Restituisce il tempo corrente in secondi dal 1° gennaio 1970 come floating point
    start_time = time.time()

    # Crea un percorso di salvataggio per il risultato degli esperimenti
    # result è la directory di base, al percorso base viene aggiunto
    # - Nome dell'esperimento
    # - E viene formattato il tempo da millisecondi a mesi, giorni, ora, minuti e secondi
    save_path = 'result' + r'/expe_[{}]_{}'.format(args.experiment_name,
                                                   time.strftime('%m-%d-%H-%M-%S', time.localtime(start_time)))


    # Se non esiste ancora il path, allora viene creato
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # Itera sugli attributi args dati in ingresso e stampa gli argomenti
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))  # str, arg_type

    # Viene richiamata la funzione save_args che presumibilmente
    # salva gli argomenti nel path creato precedentemente nel file args.txt
    util.save_args(args, save_path)

    raw_datasets = util.init_dataset(args.dataset_name, args.dataset_config_name, args.debug)

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features

    text_column_name = "tokens"
    label_column_name = "ner_tags"

    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)

    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = util.get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    # Carica il modello pre-addestrato e il tokenizzatore
    if args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, num_labels=num_labels, trust_remote_code=args.trust_remote_code, output_hidden_states=True
        )

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code,
        padding=True, return_tensor='pt', max_length=args.max_length, truncation=True
    ) #FIXME vocab_size

    if args.model_name_or_path:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            trust_remote_code=args.trust_remote_code
        )

    # Ridimensioniamo gli embeddings solo quando necessario per evitare errori di indice
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Modello ha le label allora usiamo quelle
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if sorted(model.config.label2id.keys()) == sorted(label_list):
            # Riorganizza 'label_list' per far corrispondere l'ordinamento del modello
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            Warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(model.config.label2id.keys())}, dataset labels:"
                f" {sorted(label_list)}.\nIgnoring the model labels as a result.",
            )
    # Set the correspondences label/ID inside the model config
    # model.config.label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6,
    #  'B-MISC': 7, 'I-MISC': 8}
    # model.config.id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC',
    #  7: 'B-MISC', 8: 'I-MISC'}
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = dict(enumerate(label_list))

    # Map that sends B-Xxx label to its I-Xxx counterpart
    # b_to_i_label = [0, 2, 2, 4, 4, 6, 6, 8, 8]
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        #[idx, label]= [(0, "O"),(1, "B-PER"),(2, "I-PER"),(3, "B-ORG"), (4, "I-ORG"), (5, "B-LOC"), (6, "I-LOC"), (7, "B-MISC"), (8, "I-MISC")]
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Pre-processing the datasets
    padding = "max_length" if args.pad_to_max_length else False

    processed_raw_datasets = raw_datasets.map(
        util.tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
        fn_kwargs={"text_column_name": text_column_name,
                   "label_column_name": label_column_name,
                   "max_length": args.max_length,
                   "padding": padding,
                   "tokenizer": tokenizer,
                   "label_to_id": label_to_id,
                   "b_to_i_label": b_to_i_label,
                   "label_all_tokens": args.label_all_tokens}
    )
    #"label_all_tokens": args.label_all_tokens
    train_dataset = processed_raw_datasets["train"]
    test_dataset = processed_raw_datasets["test"]

    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=None
    )

    train_dataloader = DataLoader(
        train_dataset, collate_fn=data_collator, batch_size=args.batch_size_train
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=args.batch_size_test
    )
    model.save_pretrained("checkpoint_iniziale")

    #vocabulary_embeds = util.get_vocabulary_embeds(tokenizer, model, device, args.max_length)# Cache for embeddings

    for er in range(args.experiment_rounds):
        print("======= Repeat {} ========".format(er))
        # ========================= START TRAIN ==============================
        model = AutoModelForTokenClassification.from_pretrained("checkpoint_iniziale")
        model = model.to(device)

        #criterion = util.init_loss(args.model_name_or_path)
        #criterion = util.cross_entropy_for_onehot
        criterion = util.CustomCrossEntropyLoss()

        # (Registro)
        # Lista per memorizzare i dati di ground truth
        #gt_data_list = []
        # Lista per memorizzare le etichette di ground truth
        #gt_label_list = []
        # Lista per memorizzare i valori delle loss durante l'addestramento
        true_loss_list = []
        #gt_onehot_label = util.label_to_onehot(label_list, num_labels)
        # Imposta il modello in modalità di addestramento
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # Ciclo per un numero d'iterazioni stabilite inizialmente
        for epoch in range(args.num_train_epochs):
            # Stampa l'iterazione corrente del ciclo
            print("------Epoch {}------".format(epoch))
            for iter, batch in enumerate(train_dataloader):
                gt_onehot_label_batch = []
                for sentence in batch['labels'].tolist():
                    gt_onehot_label_batch += [util.label_to_onehot(torch.tensor(sentence), num_labels).tolist()]
                gt_onehot_label_batch = torch.tensor(gt_onehot_label_batch).to(device)
                optimizer.zero_grad()
                #gt_data_list.append(batch['input_ids'].tolist())
                #gt_label_list.append(batch['labels'].tolist())
                batch.to(device)
                pred = model(batch['input_ids'], batch['attention_mask'])

                #loss = criterion(pred.logits, torch.tensor(gt_onehot_label_batch).to(device))
                loss = criterion(pred.logits, gt_onehot_label_batch).to(device)

                # Aggiunge il valore della loss alla lista delle loss
                true_loss_list.append(loss.item())

                # if loss.item() < 1e-6 or args.global_iterations < iter:
                #   break
                if args.global_iterations < iter:
                    break

                # Calcola il gradiente rispetto ai parametri del modello
                # Questo è il passo principale dell'ottimizzazione, poiché i gradienti
                # verranno utilizzati per aggiornare i pesi del modello durante l'ottimizzazione
                dy_dx = torch.autograd.grad(loss, model.parameters(), retain_graph=True)

                # =========================================== DP-SGD ============================================ #
                # Implementazione Differentially Private Stochastic Gradient Descent
                if args.is_DP:
                    # Itera sui gradienti calcolati rispetto ai parametri del modello
                    for d in dy_dx:
                        # Calcola la norma 2 del gradiente (passo utilizzato per misurare
                        # la grandezza complessiva del gradiente)
                        norm2 = torch.norm(d, p=2)

                        # Normalizza il gradiente in base al parametro di clipping
                        # (passo comune in DP-SGD per limitare la sensibilità del gradiente)
                        d.data = d.data / max(1, norm2 / args.dp_C)

                        # Crea un tensore di zeri con la stessa forma del gradiente
                        mean = torch.zeros(d.shape)

                        # Calcola la deviazione standard del rumore differenziale
                        # da aggiungere al gradiente
                        std = math.sqrt(2 * math.log(1.25 / args.dp_delta)) / args.dp_epsilon * args.dp_C

                        # Genera rumore differenziale utilizzando una distribuzione normale
                        noise = torch.normal(mean, std).to(device)

                        # Aggiunge il rumore differenziale al gradiente normalizzato
                        # (Rende il processo di addestramento differenzialmente privato)
                        d.data = d.data + noise
                # ============================================================================================= #

                # =========================================== dlg attack =========================================== #
                # Esegue un attacco di leakage durante l'addestramento
                if args.is_dlg == 1 and iter % args.dlg_attack_interval == 0:
                    attack_record = dlg_attack.dlg_attack(args,
                                               batch=batch,
                                               batch_size=args.batch_size_train,
                                               model=copy.deepcopy(model),
                                               true_dy_dx=dy_dx,
                                               dlg_attack_round=args.dlg_attack_rounds,
                                               dlg_iteration=args.dlg_iterations,
                                               dlg_lr=args.dlg_lr,
                                               epoch=epoch,
                                               global_iter=iter,
                                               model_name=args.model_name_or_path,
                                               gt_data=batch['input_ids'].tolist(),
                                               gt_label=batch['labels'].tolist(),
                                               save_path=save_path,
                                               device=device,
                                               dataset=train_dataloader,
                                               num_labels=num_labels,
                                               er=er,
                                               max_length=args.max_length,
                                               tokenizer=tokenizer,
                                               alpha=args.alpha)
                # ================================================================================================== #
                # Itera attraverso i parametri del modello ('server_param') e i gradienti calcolati ('grad_param')
                # Ogni parametro del modello viene associato al suo corrispettivo gradiente calcolato

                loss.backward()
                optimizer.step()

                print("Iter {} origin task: true_loss:{}".format(iter, loss))

        #pickle.dump(gt_data_list, open(save_path + '/gt_data_list_er={}.pickle'.format(er), 'wb'))
        #pickle.dump(gt_label_list, open(save_path + '/gt_label_list_er={}.pickle'.format(er), 'wb'))
        pickle.dump(true_loss_list, open(save_path + '/true_loss_list_er={}.pickle'.format(er), 'wb'))
        print("Finish Train!")

        # ====================== START TEST ==========================
        print("Start Test!")
        # Libera la memoria
        del train_dataloader
        torch.cuda.empty_cache()
        model.eval()
        # Lista per memorizzare i dati durante il test
        test_data_list = []
        # Lista per memorizzare le etichette di ground truth durante il test
        test_label_list = []
        # Lista per memorizzare le predizioni del modello durante il test
        test_pred_list = []
        list_pred = []
        list_real = []
        for iter, batch in enumerate(test_dataloader):
            #test_data_list.append(batch['input_ids'].tolist()[0])
            #test_label_list.append(batch['labels'].tolist()[0])
            batch.to(device)
            pred = model(**batch)
            tmp_pred = util.encode_labels_ids(pred.logits)
            for real_label, pred_label in zip(batch['labels'].tolist()[0], tmp_pred):
                if real_label != -100:
                    list_real.append(real_label)
                    list_pred.append(pred_label)

        metrics_dict = {'accuracy': metrics.accuracy_score(list_real, list_pred),
                        'f1': metrics.f1_score(list_real, list_pred, average=None),
                        'precision': metrics.precision_score(list_real, list_pred, average=None),
                        'recall': metrics.recall_score(list_real, list_pred, average=None)}

        pickle.dump(metrics_dict, open(save_path + '/measure_dict_er={}.pickle'.format(er), 'wb'))
        print("Finished Test!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="test", help="the name of experiment")
    parser.add_argument("--dataset_name", type=str, default='conll2003', help="dataset")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--user_id", type=int, default=1, help="user ID")
    parser.add_argument("--batch_size_train", type=int, default=1, help="the size of data used in training")
    parser.add_argument("--batch_size_test", type=int, default=1, help="the size of data used in test")
    parser.add_argument("--model_name_or_path", type=str, default="distilbert/distilbert-base-uncased", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Initial learning rate (after the potential warmup perdiod) to use")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform")
    parser.add_argument("--debug", action="store_true", help="Activate debug mode and run trianing only with a subset of data")
    parser.add_argument("--experiment_rounds", type=int, default=1, help="the rounds of experiments")
    parser.add_argument("--global_iterations", type=int, default=2000, help="the global iterations for original task")
    parser.add_argument(
        "--max_length",
        type=int,
        default=16,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        default=True,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    # param for DLG
    parser.add_argument("--is_dlg", type=int, default=1, help="enable DLG")
    parser.add_argument("--dlg_attack_interval", type=int, default=500, help="the interval between two dlg attack")
    parser.add_argument("--dlg_attack_rounds", type=int, default=1, help="the rounds of dlg attack")
    parser.add_argument("--dlg_iterations", type=int, default=200, help="the iterations for dlg attack")
    parser.add_argument("--dlg_lr", type=float, default=0.05, help="learning rate for dlg attack")
    parser.add_argument("--alpha", type=float, default=10, help="alpha used to calculate gradient distance")

    # param for DP-SGD
    parser.add_argument("--is_DP", type=int, default=0, help="if use DP")
    parser.add_argument("--dp_C", type=float, default=0.2, help="clipping threshold in DP")
    parser.add_argument("--dp_epsilon", type=float, default=7, help="epsilon in DP")
    parser.add_argument("--dp_delta", type=float, default=1e-5, help="delta in DP")

    args = parser.parse_args()

    main(args)