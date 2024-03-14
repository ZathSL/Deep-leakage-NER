python main.py `
--experiment_name exp1_0.05_alpha_5 `
--model_name_or_path google-bert/bert-base-uncased `
--learning_rate 0.05 `
--num_train_epochs 1 `
--experiment_rounds 1 `
--global_iterations 1500 `
--max_length 16 `
--is_dlg 1 `
--dlg_attack_interval 25 `
--dlg_attack_rounds 1 `
--dlg_iterations 1000 `
--dlg_lr 0.05 `
--alpha 5 `
--is_DP 0 `
--dp_C 0.0 `
--dp_epsilon 0 `
--dp_delta 0


distilbert/distilbert-base-uncased