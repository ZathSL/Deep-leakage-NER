import copy
import random

import torch.nn
from datasets import load_dataset
from torch import cosine_similarity
import concurrent.futures

def save_args(args, save_path):
    file = open(save_path + r'\args.txt', 'w')
    for arg in vars(args):
        file.write("{}==>{}\n".format(arg, str(getattr(args, arg))))
    file.close()


def init_dataset(dataset_name, dataset_config_name, debug):
    if dataset_name is not None:
        raw_datasets = load_dataset(dataset_name, dataset_config_name)

    if debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))

    return raw_datasets


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def tokenize_and_align_labels(examples, text_column_name,
                              label_column_name, max_length,
                              padding, tokenizer, label_to_id, b_to_i_label, label_all_tokens):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        max_length=max_length,
        padding=padding,
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # I token speciali hanno l'identificatore di parola None.
            # Definiamo l'etichetta -100 così automaticamente
            # vengono ignorati dalla funzione di loss
            if word_idx is None:
                label_ids.append(-100) # -100
            # Inseriamo l'etichetta per il primo token di ogni parola
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # Per gli atrli token nella parola, inseriamo l'etichetta sia alla corrente etichetta o -100,
            else:
                if label_all_tokens:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    label_ids.append(-100) # -100
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def init_loss(model_name):
    if 'bert' in model_name:
        loss = torch.nn.CrossEntropyLoss(reduction="mean")
    else:
        raise ValueError("No such model: " + model_name + "!")
    return loss


def train_for_loss(args, model_name, model, loss_fn, batch_size, gt_data, gt_label, global_iter, dataset, device):
    model.train()
    if 'bert' in model_name:
        gt_data_final = gt_data.copy()
        gt_label_final = gt_label.copy()


def init_dummy_data(batch_size, model, max_length, device, num_labels, tokenizer, true_dy_dx):
    sentence = []
    for i in range(batch_size):
        text = ''
        for _ in range(max_length):
            indice_parola_random = random.randint(0, len(tokenizer.vocab) - 1)
            parola_random = tokenizer.convert_ids_to_tokens(indice_parola_random)
            if '#' in parola_random:
                text += 'hello' + ' '
            else:
                text += parola_random + ' '
        sentence.append(text)
    encoding = tokenizer(
        sentence,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )
    with torch.no_grad():
        dummy_labels = torch.nn.Parameter(torch.rand((batch_size, max_length, num_labels)) * (num_labels - 1))
        encoding['labels'] = dummy_labels
        encoding.to(device)
    return encoding


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


# def cross_entropy_for_onehot(pred, target):
#    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))




def find_best_match(args):
    embed_token, vocabulary_embeds = args
    max_sim = 0
    better_word = ''
    for word, sentence_encoding in vocabulary_embeds.items():
        similarity_score = cosine_similarity(embed_token.unsqueeze(0), sentence_encoding).item()
        if similarity_score > max_sim:
            better_word = word
            max_sim = similarity_score
    return better_word


def convert_emb_to_text_parallel(token_embeds_dummy, tokenizer, model, device, vocabulary_embeds, num_threads=8):
    batch_sentences = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for batch in token_embeds_dummy:
            sentence = list(executor.map(find_best_match, [(token, vocabulary_embeds) for token in batch]))
            decoded_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(sentence), skip_special_tokens=True)
            batch_sentences.append(decoded_text)
    return batch_sentences


def convert_emb_to_text(token_embeds_dummy, tokenizer, model, device, vocabulary_embeds):
    batch_sentences = []
    # Process batches using parallelism
    for batch in token_embeds_dummy:
        sentence = []

        # Parallel processing of each token in the batch
        for embed_token in batch:
            max_sim = 0
            better_word = ''

            # Compare with precomputed embeddings in the vocabulary
            for word, sentence_encoding in vocabulary_embeds.items():
                similarity_score = cosine_similarity(embed_token.unsqueeze(0), sentence_encoding).item()

                if similarity_score > max_sim:
                    better_word = word
                    max_sim = similarity_score

            sentence.append(better_word)
        decoded_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(sentence), skip_special_tokens=True)
        batch_sentences.append(decoded_text)

    return batch_sentences


def get_vocabulary_embeds(tokenizer, model, device, max_length):
    print(" Sto caricando gli embeddings del vocabolario ")
    vocabulary_embeds = {}
    vocabulary = tokenizer.get_vocab()
    # vocabulary = {k: vocabulary[k] for k in list(vocabulary)[:10]}
    # Precompute embeddings for the vocabulary
    for word in vocabulary.keys():
        encoding = tokenizer(
            word,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
        embedding_word = model(**encoding)['hidden_states'][0]
        sentence_encoding = embedding_word.mean(dim=1)
        vocabulary_embeds[word] = sentence_encoding.clone().detach().to(device)
    print(" Ho terminato di caricare gli embeddings del vocabolario ")
    return vocabulary_embeds


def get_embedding(word, tokenizer, model, device, max_length):
    encoding = tokenizer(
        word,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )
    embedding_word = model(**encoding)['hidden_states'][0]
    sentence_encoding = embedding_word.mean(dim=1)

    return word, sentence_encoding.clone().detach().to(device)


def get_vocabulary_embeds_parallel(tokenizer, model, device, max_length):
    print(" Sto caricando gli embeddings del vocabolario ")
    vocabulary_embeds = {}
    vocabulary = tokenizer.get_vocab()
    # vocabulary = {k: vocabulary[k] for k in list(vocabulary)[:10]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_word = {executor.submit(get_embedding, word, copy.deepcopy(tokenizer), model, device, max_length): word for word in vocabulary.keys()}
        for future in concurrent.futures.as_completed(future_to_word):
            word = future_to_word[future]
            try:
                vocabulary_embeds[word] = future.result()[1]
            except Exception as exc:
                print('%r generated an exception: %s' % (word, exc))

    print(" Ho terminato di caricare gli embeddings del vocabolario ")
    return vocabulary_embeds


def encode_labels(labels_encoding, model):
    batch_labels = []
    for batch in labels_encoding:
        labels_tokens = []
        for token_pred in batch:
            probabilities = torch.nn.functional.softmax(token_pred, dim=0)
            label_index = torch.argmax(probabilities)
            labels_tokens.append(model.config.id2label[label_index.item()])
        batch_labels.append(labels_tokens)

    return batch_labels


def encode_labels_ids(labels_encoding):
    batch_labels = []
    for batch in labels_encoding:
        for token_pred in batch[0]:
            probabilities = torch.nn.functional.softmax(token_pred, dim=0)
            label_index = torch.argmax(probabilities)
            batch_labels.append(label_index.item())
    return batch_labels
