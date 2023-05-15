from torch.utils.data import DataLoader

def generate_nmt_batches(dataset, batch_size, shuffle=True, drop_last = True, device="cpu"):
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=shuffle, drop_last = drop_last)

    for data_dict in dataloader:
        lengths = data_dict['x_source_legnth'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].toList()

        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict