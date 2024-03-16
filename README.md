python main.py `
--experiment_name distilbert_alpha_15_ `
--model_name_or_path distilbert/distilbert-base-uncased `
--learning_rate 0.05 `
--num_train_epochs 1 `
--experiment_rounds 1 `
--global_iterations 1500 `
--max_length 16 `
--is_dlg 1 `
--dlg_attack_interval 300 `
--dlg_attack_rounds 2 `
--dlg_iterations 1000 `
--dlg_lr 0.05 `
--alpha 15 `
--is_DP 0 `
--dp_C 0.0 `
--dp_epsilon 0 `
--dp_delta 0


!python main.py --experiment_name bert_alpha_10_dp_0_lr_0.05_len_16 \
--model_name_or_path google-bert/bert-base-uncased \
--learning_rate 0.05 \
--num_train_epochs 1 \
--experiment_rounds 1 \
--global_iterations 1500 \
--max_length 16 \
--is_dlg 1 \
--dlg_attack_interval 300 \
--dlg_attack_rounds 2 \
--dlg_iterations 500 \
--dlg_lr 0.05 \
--alpha 10 \
--is_DP 0 \
--dp_C 0.0 \
--dp_epsilon 0 \
--dp_delta 0


distilbert/distilbert-base-uncased