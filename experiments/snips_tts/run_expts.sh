############################## Train on CLEAN ###################################
python3 experiments/snips_tts/trainer_vanilla.py --use_clean
python3 experiments/snips_tts/trainer_vanilla.py --use_clean
python3 experiments/snips_tts/trainer_vanilla.py --use_clean
python3 experiments/snips_tts/trainer_vanilla.py --use_clean
python3 experiments/snips_tts/trainer_vanilla.py --use_clean

############################## Train on True ASR ################################
python3 experiments/snips_tts/trainer_vanilla.py
python3 experiments/snips_tts/trainer_vanilla.py
python3 experiments/snips_tts/trainer_vanilla.py
python3 experiments/snips_tts/trainer_vanilla.py
python3 experiments/snips_tts/trainer_vanilla.py

############################## Train on True ASR 5-Best Hypotheses ##############
python3 experiments/snips_tts/trainer_vanilla.py --use_n_best
python3 experiments/snips_tts/trainer_vanilla.py --use_n_best
python3 experiments/snips_tts/trainer_vanilla.py --use_n_best
python3 experiments/snips_tts/trainer_vanilla.py --use_n_best
python3 experiments/snips_tts/trainer_vanilla.py --use_n_best

