import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

def prepare_dataloader(tokenizer, dataset, batch_size):
    """
    Prepare data for BERT inference

    Args:
        tokenizer (BertTokenizer): tokenizer used to encode text
        dataset (pandas.DataFrame): data frame with columns "text" and "target" depicting the input data and label
        batch_size (int): the batch size that should be used
    """
    examples = list(dataset["text"])
    input_ids_list = []
    mask_list = []

    for text in examples:
        encoded_text = tokenizer.encode_plus(
            text,
            max_length=64,
            padding='max_length',
            truncation=True
        )

        input_ids_list.append(encoded_text["input_ids"])
        mask_list.append(encoded_text["attention_mask"])

    ids = torch.tensor(input_ids_list)
    mask = torch.tensor(mask_list)
    labels = torch.tensor(dataset["target"].values, dtype=torch.long)

    # Create the DataLoader for our dataset.
    data = TensorDataset(ids, mask, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader
