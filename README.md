python main.py `
--experiment_name distilbert_tuning_parameters_law_11`
--model_name_or_path distilbert/distilbert-base-uncased `
--learning_rate 0.05 `
--num_train_epochs 1 `
--experiment_rounds 1 `
--global_iterations 1 `
--max_length 16 `
--batch_size_train 1 `
--is_dlg 1 `
--dlg_attack_interval 300 `
--dlg_attack_rounds 1 `
--dlg_iterations 200 `
--dlg_lr 0.05 `
--decay_rate_alpha 0.08 `
--is_DP 0 `
--dp_C 0.0 `
--dp_epsilon 0 `
--dp_delta 0




!python main.py --experiment_name distilbert_tuning_parameters_law_05 \
--model_name_or_path distilbert/distilbert-base-uncased \
--learning_rate 0.05 \
--num_train_epochs 1 \
--experiment_rounds 1 \
--global_iterations 1 \
--max_length 16 \
--is_dlg 1 \
--dlg_attack_interval 300 \
--dlg_attack_rounds 1 \
--dlg_iterations 200 \
--dlg_lr 0.05 \
--decay_rate_alpha 0.08 \
--is_DP 0 \
--dp_C 0.0 \
--dp_epsilon 0 \
--dp_delta 0


distilbert/distilbert-base-uncased