from torch.cuda import device_count

def get_number_of_gpus():
    return device_count()
