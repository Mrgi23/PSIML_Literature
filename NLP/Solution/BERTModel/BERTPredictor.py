import torch
import numpy as np
import torch.nn.functional as F


def predict(text, model, tokenizer):
    """
    Model inference on some dataset

    Args:
        text (str): the text of the tweet
        model (BertForSequenceClassification): BERT model with additional classification layer 
        tokenizer (BertTokenizer): a tokenizer used to encode text
    Return:
        int: predicted sentiment of the tweet
    """
    predicted_target = 0

    # TODO WORKSHOP TASK
    # 1. put model in evaluation mode
    # 2. encode text using the Tokenizer
    # 3. transform inputs and mask to (long) tensor
    # 4. model forward pass
    # 5. fetch logits nad calculate softmax outputs
    # 6. calculate prediction



    # END WORKSHOP TASK

    return predicted_target
