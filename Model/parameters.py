import torch

def  GetParameters():
    # parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    learning_rate = 1e-3
    batch_size = 8

    day_num = 365

    seq_len = 14
    pre_len = 7
    hidden_size = 256

    envs_width = 80
    envs_heigh = 80
    heat_width = 100
    heat_heigh = 100
    envs_feature_len = 400
    heat_feature_len = 2500
    Layer = 6

    parameters = dict(device=device,
                      learning_rate=learning_rate,
                      batch_size=batch_size,
                      day_num=day_num,
                      seq_len=seq_len,
                      pre_len=pre_len,
                      hidden_size=hidden_size,
                      envs_width = envs_width,
                      envs_heigh = envs_heigh,
                      heat_width = heat_width,
                      heat_heigh = heat_heigh,
                      envs_feature_len = envs_feature_len,
                      heat_feature_len = heat_feature_len,
                      Layer = Layer
                      )
    return parameters
