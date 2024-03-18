import pickle
import re

import torch
from sklearn import metrics
from torch import cosine_similarity, optim
from tqdm import tqdm
import util
from rouge import Rouge
import torch.nn.functional as F
import subprocess


def find_parameter(args, batch, batch_size, model, true_dy_dx, dlg_attack_round, dlg_iteration, dlg_lr, epoch, global_iter, model_name,
               gt_data, gt_label, save_path, device, dataset, num_labels, er, max_length, tokenizer, decay_rate_alpha):
    # Inizializza lista per registrare i risultati dell'attacco
    attack_record_list = list()

    # Inizializza una funzione loss per il modello adottato
    criterion = util.CustomCrossEntropyLoss()
    #criterion = util.init_loss(model_name)
    #criterion = util.cross_entropy_for_onehot

    # Inizializza la distanza euclidea utilizzando il modulo PyTorch con norma 2
    edist2 = torch.nn.PairwiseDistance(p=2)
    edist1 = torch.nn.PairwiseDistance(p=1)
    
    destination_path = "/content/drive/MyDrive/LM"
    
    model.train()
    law = 1.1
    while law < 1.2:
        decay = 0.1
        # Esecuzione dell'attacco DLG per un numero specificato di round
        while decay < 25:
            # Dizionario per registrare i risultati dell'attacco corrente
            attack_record = {'grad_loss': [], 'dummy_data_list': [], 'dummy_label_list': [], 'last_dummy_data': [], 'dlg_iteration': [],
                             'last_dummy_label': [], 'epoch': epoch, 'global_iteration': global_iter, 'model_name': model_name,
                             'gt_data': [], 'gt_label': [], 'cosine_similarity_data': []}

            # Decadimento di alpha tramite la legge di potenza
            initial_alpha = alpha = 1
            # Tasso di decadimento (più basso decadimento più lento)
            decay_rate = decay
            # Decadimento in base al numero di iterazioni, 1 implica decadimento lineare
            power = law
            scheduler = util.PowerDecayScheduler(initial_alpha, decay_rate, power)

            # Memorizza l'input di training da rubare
            for data in gt_data:
                attack_record['gt_data'].append(tokenizer.decode(data))

            # Memorizza le label di training da rubare
            for labels in gt_label:
                list_tmp = []
                for label in labels:
                    if label == -100:
                        list_tmp.append(label)
                    else:
                        list_tmp.append(model.config.id2label[label])
                attack_record['gt_label'].append(list_tmp)

            # Inizializzazione dei dati e delle etichette dummy utilizzati durante l'attacco
            sentence_dummy, dummy_labels = util.init_dummy_data2(batch_size=batch_size, model=model, max_length=max_length, device=device,num_labels=num_labels, tokenizer=tokenizer, true_dy_dx=true_dy_dx)

            # Estrai l'embedding del modello
            embedding = model.get_input_embeddings()

            # Calcola i token embedding dummy e reale
            with torch.no_grad():
                token_embeds_dummy = embedding(sentence_dummy)
                token_embeds_real = embedding(batch['input_ids'])

            # Richiedi i gradienti per ottimizzare
            token_embeds_dummy = torch.nn.Parameter(token_embeds_dummy, requires_grad=True)
            dummy_labels = torch.nn.Parameter(dummy_labels, requires_grad=True)

            optimizer = optim.Adam([token_embeds_dummy, dummy_labels], lr=dlg_lr)

            # Esegue l'ottimizzazione SGD per un numero specificato di iterazioni
            for c in tqdm(range(dlg_iteration)):
                ######################## ATTACK ###################################################
                def closure():
                    nonlocal token_embeds_dummy, dummy_labels, alpha
                    dummy_pred = model(inputs_embeds=token_embeds_dummy)

                    # Calcola la loss dummy
                    loss = criterion(dummy_pred.logits, dummy_labels)
                    #util.cross(dummy_pred.logits, dummy_labels)

                    # Calcola i gradienti dummy
                    dummy_dl_dw = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)

                    grad_distance = []
                    pattern = r'\d+\.?\d*'
                    tmp = 'embedding'

                    # Stampa i nomi dei parametri e i relativi gradienti
                    for (param_name, _), gradient_dummy, gradient_real in zip(model.named_parameters(), dummy_dl_dw, true_dy_dx):
                        if tmp not in param_name:
                            #alpha *= 0.6
                            scheduler.step()
                            alpha = scheduler.get_alpha()
                            if 'embedding' in param_name:
                                tmp = 'embedding'
                            elif re.findall(pattern, param_name):
                                tmp = str(float(re.findall(pattern, param_name)[0]).__round__())
                            elif 'classifier' in param_name:
                                tmp = 'classifier'


                        if 'embeddings.word_embeddings.weight' in param_name:
                            continue
                        else:
                            result_e2 = edist2(gradient_dummy, gradient_real)
                            result_e1 = edist1(gradient_dummy, gradient_real)

                        grad_distance.append(result_e2 + alpha * result_e1)

                    partial_sum = []
                    for i in grad_distance:
                        partial_sum.append(torch.sum(i))
                    grad_distance_mean = torch.mean(torch.stack(partial_sum))
                    # Passo di ottimizzazione nei dati e label dummy
                    optimizer.zero_grad()
                    grad_distance_mean.backward(retain_graph=True)
                    optimizer.step()

                    return grad_distance_mean
                ########################################################################################
                grad_distance_sum = closure()

            batch_sentences = util.convert_embeddings_to_text(token_embeds_dummy, embedding.weight.data, tokenizer)
            batch_labels = util.encode_labels(dummy_labels, model)
            attack_record['last_dummy_data'] = batch_sentences
            attack_record['last_dummy_label'] = batch_labels
            attack_record['grad_loss'] = grad_distance_sum.item()
            attack_record['last_cosine_similarity_data'] = util.calculate_similarity(token_embeds_dummy, token_embeds_real)

            # attack_record_list.append(attack_record)
            str_path = save_path+'/attack_record_gloiter={}_power={}_decay_rate={}.pickle'.format(global_iter, law, decay)
            pickle.dump(attack_record, open(str_path, 'wb'))
            command = ["cp", "-r", str_path, destination_path]

            try:
                # Esegui il comando e acquisisci l'output
                output = subprocess.check_output(command)
                
                # Decodifica l'output dai byte in una stringa
                output_str = output.decode("utf-8")
                
                # Stampa l'output
                print(output_str)
            except subprocess.CalledProcessError as e:
                # Se il comando non ha successo, cattura l'eccezione e stampa un messaggio di errore
                print("Errore nell'esecuzione del comando:", e)
            
            

            list_pred = []
            for batch_i in attack_record['last_dummy_label']:
                for label in batch_i:
                    list_pred.append(model.config.label2id[label])

            list_real = []
            for sentence in gt_label:
                for token in sentence:
                    list_real.append(token)
            list_real_final = []
            list_pred_final = []
            for pred, real in zip(list_pred, list_real):
                if real != -100:
                    list_pred_final.append(pred)
                    list_real_final.append(real)
            del(list_real, list_pred)
            # Calcola le metriche in riferimento al recupero delle label
            metrics_dict = {'accuracy': metrics.accuracy_score(list_real_final, list_pred_final),
                            'f1': metrics.f1_score(list_real_final, list_pred_final, average='micro'),
                            'precision': metrics.precision_score(list_real_final, list_pred_final, average='micro'),
                            'recall': metrics.recall_score(list_real_final, list_pred_final, average='micro')}

            # Recovery rate & Rouge in riferimento al recupero dei dati di training
            rouge = Rouge()
            for i, (sentence_real, sentence_dummy) in enumerate(zip(attack_record['gt_data'], attack_record['last_dummy_data'])):
                sentence_real = re.sub(r'[^\w\s]', '', sentence_real)
                sentence_dummy = re.sub(r'[^\w\s]', '', sentence_dummy)
                words_dummy = sentence_dummy.split()
                words_real = sentence_real.split()
                len_real = len(words_real)
                recovery = 0
                for word in words_dummy:
                    try:
                        words_real.remove(word)
                        recovery += 1
                    except ValueError:
                        pass
                scores = rouge.get_scores(sentence_real, sentence_dummy, avg=None)
                metrics_dict[str(i) + ' ' + 'rouge-1'] = scores[0]['rouge-1']
                metrics_dict[str(i) + ' ' + 'rouge-2'] = scores[0]['rouge-2']
                metrics_dict[str(i) + ' ' + 'rouge-l'] = scores[0]['rouge-l']

                # Calcola la percentuale di recupero del testo
                metrics_dict[str(i) + ' ' + 'recovery_rate'] = recovery / len_real
            
            str_path = save_path + '/measure_leakage_gloiter={}_power={}_decay_rate={}.pickle'.format(global_iter, law, decay)
            pickle.dump(metrics_dict, open(str_path, 'wb'))
            command = ["cp", "-r", str_path, destination_path]
            try:
                # Esegui il comando e acquisisci l'output
                output = subprocess.check_output(command)
                
                # Decodifica l'output dai byte in una stringa
                output_str = output.decode("utf-8")
                
                # Stampa l'output
                print(output_str)
            except subprocess.CalledProcessError as e:
                # Se il comando non ha successo, cattura l'eccezione e stampa un messaggio di errore
                print("Errore nell'esecuzione del comando:", e)
            
            decay += 0.1
            decay = round(decay, 1)
        law += 0.1
        law = round(law, 1)

    return attack_record_list

