import torch
import torch.nn as nn

__all__ = ['LSTM']


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                num_layers=2):
        """
        Vanilla Stacked LSTM Block
        Currently only outputs the final element in the sequence.

        Parameters
        ----------
        input_dim : int
            Dimension of the input.
        hidden_dim : int
            Dimension of the hidden layer.
        batch_size : int
            Size of the batch.
        output_dim : int, optional
            Dimension of the input, by default 1
        num_layers : int, optional
            Number of LSTM Cells in the block , by default 2
        """
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = torch.nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        """
        Hidden State initialisation.

        Returns
        -------
        Hidden State : The Hidden State and the Cell State.
            [description]
        """
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        """
        Forward Pass

        Parameters
        ----------
        input : torch.Tensor
            Intput to the LSTM Block.

        Returns
        -------
        output : torch.Tensor
            Output of the LSTM Block.
        """
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input),
                                          self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred
