import pickle
import re

import torch
from sklearn import metrics
from torch import cosine_similarity, optim
from tqdm import tqdm
import util
from rouge import Rouge
import torch.nn.functional as F


def dlg_attack(args, batch, batch_size, model, true_dy_dx, dlg_attack_round, dlg_iteration, dlg_lr, epoch, global_iter, model_name,
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

    model.train()

    # Esecuzione dell'attacco DLG per un numero specificato di round
    for r in range(dlg_attack_round):
        # Dizionario per registrare i risultati dell'attacco corrente
        attack_record = {'grad_loss_list': [], 'dummy_data_list': [], 'dummy_label_list': [], 'last_dummy_data': [], 'dlg_iteration': [],
                         'last_dummy_label': [], 'epoch': epoch, 'global_iteration': global_iter, 'model_name': model_name,
                         'gt_data': [], 'gt_label': [], 'cosine_similarity_data': []}

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
                nonlocal token_embeds_dummy, dummy_labels
                dummy_pred = model(inputs_embeds=token_embeds_dummy)

                # Calcola la loss dummy
                loss = criterion(dummy_pred.logits, dummy_labels)
                #util.cross(dummy_pred.logits, dummy_labels)

                # Calcola i gradienti dummy
                dummy_dl_dw = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)

                grad_distance = []
                pattern = r'\d+\.?\d*'
                tmp = 'embedding'
                # Decadimento di alpha tramite la legge di potenza
                initial_alpha = alpha = 1
                # Tasso di decadimento (più basso decadimento più lento)
                decay_rate = 10
                # Decadimento in base al numero di iterazioni, 1 implica decadimento lineare
                power = 5
                scheduler = util.PowerDecayScheduler(initial_alpha, decay_rate, power)

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
                    elif 'embeddings' in param_name:
                        result_e2 = edist2(gradient_dummy, gradient_real)
                        result_e1 = edist1(gradient_dummy, gradient_real)
                    elif 'layer.' in param_name:
                        result_e2 = edist2(gradient_dummy, gradient_real)
                        result_e1 = edist1(gradient_dummy, gradient_real)
                        #alpha *= 0.4 #0.9 e 0.7 it's ok
                        #scheduler.step()
                        #alpha = scheduler.get_alpha()
                    else: # 'classifier' or 'embedding_position'
                        result_e2 = edist2(gradient_dummy, gradient_real)
                        result_e1 = edist1(gradient_dummy, gradient_real)
                        #alpha *= 0.4
                        #scheduler.step()
                        #alpha = scheduler.get_alpha()

                    grad_distance.append(result_e2 + alpha * result_e1)

                partial_sum = []
                for i in grad_distance:
                    partial_sum.append(torch.sum(i))
                grad_distance_mean = torch.mean(torch.stack(partial_sum))
                # Passo di ottimizzazione nei dati e label dummy
                optimizer.zero_grad()
                grad_distance_mean.backward(retain_graph=True)
                optimizer.step()
                #grad = torch.autograd.grad(grad_distance_sum, [token_embeds_dummy, dummy_labels], retain_graph=True)

                #token_embeds_dummy = token_embeds_dummy - dlg_lr * grad[0]
                #dummy_labels = dummy_labels - dlg_lr * grad[1]

                #probabilities = F.softmax(dummy_labels, dim=2)
                #predicted_classes = torch.argmax(probabilities, dim=2)
                #dummy_onehot_label = []
                #for sentence in predicted_classes.tolist():
                #    dummy_onehot_label += [util.label_to_onehot(torch.tensor(sentence), num_labels).tolist()]
                #dummy_onehot_label = torch.tensor(dummy_onehot_label).to(device).requires_grad_(True)
                #print(dummy_labels)
                #dummy_labels = F.softmax(dummy_labels, dim=2)

                return grad_distance_mean
            ########################################################################################
            grad_distance_sum = closure()

            # Salvataggio checkpoint attacco
            if c % 20 == 0:
                batch_sentences = util.convert_embeddings_to_text(token_embeds_dummy, embedding.weight.data, tokenizer)
                batch_labels = util.encode_labels(labels_encoding=dummy_labels, model=model)
                # Aggiunge i dati dummy alla lista di dati dummy
                attack_record['dummy_data_list'].append(batch_sentences)
                attack_record['grad_loss_list'].append(grad_distance_sum.item())
                # Aggiunge le etichette dummy alla lista delle etichette dummy
                attack_record['dummy_label_list'].append(batch_labels)
                # Calcola la similarità del coseno tra il dato reale ed il dato dummy ottimizzato
                cos_sim = util.calculate_similarity(token_embeds_dummy, token_embeds_real)
                attack_record['cosine_similarity_data'].append(cos_sim)
                print(grad_distance_sum, "\n")
                print(batch_sentences, "\n")
                print(attack_record['gt_data'], "\n")
                print(batch_labels, "\n")
                print(attack_record['gt_label'], "\n")
                del batch_sentences, batch_labels, cos_sim, grad_distance_sum

        batch_sentences = util.convert_embeddings_to_text(token_embeds_dummy, embedding.weight.data, tokenizer)
        batch_labels = util.encode_labels(dummy_labels, model)
        attack_record['last_dummy_data'] = batch_sentences
        attack_record['last_dummy_label'] = batch_labels

        attack_record['last_cosine_similarity_data'] = util.calculate_similarity(token_embeds_dummy, token_embeds_real)

        attack_record_list.append(attack_record)
        pickle.dump(attack_record, open(save_path+'/attack_record_round={}_gloiter={}.pickle'.format(r, global_iter), 'wb'))

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
        recovery_rate = []
        metrics_dict['rouge-1'] = []
        metrics_dict['rouge-2'] = []
        metrics_dict['rouge-l'] = []
        for i, (sentence_real, sentence_dummy) in enumerate(zip(attack_record['gt_data'], attack_record['last_dummy_data'])):
            recovery_rate.append(0)
            for word in sentence_real:
                if word in sentence_dummy:
                    recovery_rate[i] += 1

            scores = rouge.get_scores(sentence_real, sentence_dummy, avg=None)
            # 'r' recall, 'p' precision, 'f' f-score
            metrics_dict['rouge-1'].append(scores[0]['rouge-1'])
            metrics_dict['rouge-2'].append(scores[0]['rouge-2'])
            metrics_dict['rouge-l'].append(scores[0]['rouge-l'])
        # Calcola la percentuale di recupero del testo
        for i, rr in enumerate(recovery_rate):
            recovery_rate[i] = rr / len(gt_data[i])

        metrics_dict['recovery_rate'] = recovery_rate

        pickle.dump(metrics_dict, open(save_path + '/measure_leakage_round={}_gloiter={}.pickle'.format(r, global_iter),'wb'))

    return attack_record_list

