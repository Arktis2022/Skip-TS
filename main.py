from train_and_test import train
from eval_func import setup_seed
seeds = [42, 142, 250]

data_path = "/your/datasets/head_ct/"

if __name__ == '__main__':
    for seed in seeds:
        setup_seed(seed)
        max_auroc = train(class_ = 'head_ct',   
                    epochs = 200,
                    learning_rate = 0.005 ,
                    res = 3,      
                    batch_size = 16,
                    print_epoch = 10,
                    data_path = data_path,
                    save_path = './checkpoints/',
                    score_num = 1, 
                    print_loss = 1,
                    layerloss = 1,
                    rate = 0.05,  
                    print_max = 1,
                    net = 'wide_res50',
                    L2 = 0, 
                    seed = seed)
        print(max_auroc)
