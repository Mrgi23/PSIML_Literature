import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentClassifierTransformer(nn.Module):
    """A Transformer Encoder to extract feature representation and 2-layer multilayer perceptron to do the classification"""

    def __init__(self, embedding_size, num_embeddings, n_heads, dim_feedforward, num_layers, output_dim, pretrained_embedding_matrix=None,  padding_idx=0, batch_first=False):
        """
        Args:
            embedding_size (int): the size of the embedding vector
            num_embeddings (int): the number of words to embed
            n_heads (int): the number of heads in the multi-head attention model
            dim_feedforward (int): the dimension of the feedforward network model
            num_layers (int): the number of sub-encoder layers in the encoder
            output_dim (int): the size of the prediction vector
            pretrained_embedding_matrix (numpy.array): previously trained word embeddings
            padding_idx (int): an index in the Vocabulary representing the <MASK> token (padding)
            batch_first (bool): flag whether the batch is the 0th dimension in the input tensor
        """
        # call the base initialization
        super(SentimentClassifierTransformer, self).__init__()

        # Define the model

        if pretrained_embedding_matrix is None:
            # instantiate the Embedding layer without initial weights
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_size,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx
            )
        else:
            # firstly convert pretrained_embedding_matrix into tensor
            pretrained_embedding_matrix = torch.from_numpy(pretrained_embedding_matrix).float()

            # instantiate the Embedding layer with initial weights
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_size,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx,
                _weight=pretrained_embedding_matrix
            )

        # encoder layer is made up of self-attention and feed forward network
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=n_heads, dim_feedforward=dim_feedforward)

        # transformer encoder is a stack of encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # classification linear layer with number of inputs corresponding to the size of the embedding vector and number of outputs being equal to the number of output classes
        self.fc1 = nn.Linear(in_features=embedding_size, out_features=output_dim)


    def forward(self, x_in, x_lengths=None, apply_softmax=False):
        """
        The forward pass of the Classifier

        Args:
            x_in (torch.Tensor): an input data tensor with input shape (batch, dataset._max_sequence_length)
            x_length (torch.Tensor): the lengths of each sequence in the batch
            apply_softmax (bool): a flag for the softmax activation, which should be false if used with the cross-entropy losses
        Returns:
            resulting tensor, with shape (batch, output_dim)
        """
        # create vectors for each word in the input data tensor, by converting the indices to vectors
        x_embedded = self.embeddings(x_in.long())

        # create transformer hidden state vectors for each word in the input data tensor
        y_out = self.transformer_encoder(x_embedded)

        # calculate the transformer encoder output as the mean of all hidden state vectors
        y_out = y_out.mean(dim=1)

        # calculate the output of the classifier linear layer
        y_out = self.fc1(y_out)

        # apply softmax function to the calculate output, if needed
        if (apply_softmax):
            y_out = F.softmax(y_out, dim=1)

        return y_out
