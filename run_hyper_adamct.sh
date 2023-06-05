# Amazon Beauty
python run_hyper.py --gpu_id='0' --model=AdaMCT --dataset='Amazon_Beauty' --config_files='config/Amazon_Beauty.yaml' --params_file='hyper_adamct.test'

# Amazon Sports and Outdoors
python run_hyper.py --gpu_id='1' --model=AdaMCT --dataset='Amazon_Sports_and_Outdoors' --config_files='config/Amazon_Sports_and_Outdoors.yaml' --params_file='hyper_adamct.test'

# Amazon Toys and Games
python run_hyper.py --gpu_id='2' --model=AdaMCT --dataset='Amazon_Toys_and_Games' --config_files='config/Amazon_Toys_and_Games.yaml' --params_file='hyper_adamct.test'