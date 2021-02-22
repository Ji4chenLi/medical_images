# Copyright 2020 Petuum Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model classes for nlp models in medical report generation
"""
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Texar Library
from texar.torch import ModuleBase, HParams
from texar.torch.modules import UnidirectionalRNNEncoder,\
    WordEmbedder, BasicRNNDecoder
from texar.torch.modules.decoders.decoder_helpers \
    import GreedyEmbeddingHelper, TrainingHelper
from texar.torch.core.cell_wrappers import LSTMCell
from build_vocab import START_TOKEN, END_TOKEN


class CoAttention(ModuleBase):
    r"""It takes in as input the V visual features and the a semantic features along with the hidden
    vector of the previous time step of the sentence LSTM layer 1 in the 2 layer hierarchical LSTM

    Args:
        hparams (dict or HParams, optional): LSTMSentence hyperparameters. Missing
            hyperparameters will be set to default values. See :meth:`default_hparams` for the
            hyperparameter structure and default values.
                * visual_feature_dim (int): Dimension of visual unit
                * semantic_feature_dim (int): Dimension of semantic unit
                * hidden_size (int): Assuming hidden state and input to lstm have the same
                dimension, the hidden vector and input size of the sentence LSTM
                * batch_size (int): Batch size
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        visual_dim = self.hparams.visual_dim
        semantic_dim = self.hparams.hidden_size
        hidden_size = self.hparams.hidden_size

        # As per the On the Automatic Generation
        # of Medical Imaging Reports paper

        # Visual attention
        # Notation from Equation 2
        self.W_v = nn.Linear(
            in_features=visual_dim, out_features=visual_dim
        )
        self.W_v_h = nn.Linear(
            in_features=hidden_size, out_features=visual_dim
        )
        self.W_v_att = nn.Linear(
            in_features=visual_dim, out_features=visual_dim
        )

        # Semantic attention
        # Notation from Equation 3
        self.W_a = nn.Linear(
            in_features=semantic_dim, out_features=semantic_dim
        )
        self.W_a_h = nn.Linear(
            in_features=semantic_dim, out_features=semantic_dim
        )
        self.W_a_att = nn.Linear(
            in_features=semantic_dim, out_features=1
        )

        # Context calculation layer
        self.W_fc = nn.Linear(
            in_features=visual_dim + semantic_dim, out_features=hidden_size)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()


    def forward(self, visual_feature, semantic_feature, sentence_state):
        # TODO: Fill in docstring: visual_alignment and semantic_alignment
        r"""

        Args:
            v (torch.Tensor): Dimension [Batch size, max_time_steps = N_v,
            hidden_state = visual_feature_dim]
            a (torch.Tensor): Dimension [Batch size, max_time_steps = N_a,
            hidden_state = semantic_feature_dim]
            sentence_state (torch.Tensor): Hidden state for sentence_lstm

        Returns:
            ctx (torch.Tensor): Joint context vector
            visual_alignments (torch.Tensor):
            semantic_alignments (torch.Tensor):
        """

        # Visual attention
        W_v = self.W_v(visual_feature)
        W_v_h = self.W_v_h(sentence_state.squeeze(1))
        visual_score = self.W_v_att(
            self.tanh(W_v + W_v_h))
        visual_align = F.softmax(visual_score, dim=1)
        visual_att = torch.mul(visual_align, visual_feature)

        # Semantic attention
        W_a = self.W_a(semantic_feature)
        W_a_h = self.W_a_h(sentence_state).unsqueeze(1)
        semantic_score = self.W_a_att(
            self.tanh(W_a_h + W_a))

        semantic_align = F.softmax(semantic_score, dim=1)
        semantic_att = torch.mul(semantic_align, semantic_feature)
        semantic_att = semantic_att.sum(1)

        # Calculate the context
        cat_att = torch.cat([visual_att, semantic_att], dim=1)
        context = self.W_fc(cat_att)

        return context, visual_align, semantic_align.squeeze(2)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            'hidden_size': 512,
            'visual_dim': 1024,
            'semantic_dim': 512,
        }

class LstmSentence(ModuleBase):
    r"""This is an implementation of 1st in the hierarchy for 2 layered hierarchical LSTM
    implementation in Texar. In this particular application the first tier takes in the input
    which here is from co-attention and outputs a hidden state vector. At each time step of the
    tier-1 LSTM, the hidden state vector is fed into the tier 2 LSTM as a state tuple to output
    a sentence.

    NOTE: Run the LSTM sentence in a for loop range [0 max time steps] till termination
    Because at each time step we calculate the visual and semantic attention (stack at the end of
    the max timesteps to find the alpha and beta)
    Output the t needed for the Word lstm
    At each time step we produce a 0 or 1 to continue or stop

    This is run at every time step of Sentence LSTM as stated above
    It is Bernoulli variable
    p_pred shape [Batch size, 2]
    p_target shape [Batch size]

    Args:
        hparams (dict or HParams, optional): LstmSentence hyperparameters. Missing
            hyperparameters will be set to default values. See :meth:`default_hparams` for the
            hyperparameter structure and default values.
                * hidden_size (int): Hidden_size the same as GNN output
                * visual_feature_dim (int): Dimension of visual features
                * semantic_feature_dim (int): Dimension of semantic features
                * batch_size (int): Batch size
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        # Dimension of visual and semantic features
        visual_dim = self.hparams.visual_dim
        semantic_dim = self.hparams.hidden_size

        # Word embedding parameters
        self.num_tags = self.hparams.num_tags
        self.embed = nn.Embedding(self.num_tags + 1, semantic_dim)
        self.k = self.hparams.top_k_for_semantic
        self.threshold = self.hparams.threshold

        # LSTM parameters
        # the hidden vector and input size of the sentence LSTM
        self.hidden_size = self.hparams.hidden_size

        # The Co_Attention module
        # Observe that the input to the LSTM and
        # hidden vector have the same dimension
        # If not add a new parameter and make changes accordingly
        hparams_coattn = {
            "visual_dim": visual_dim,
            "semantic_dim": semantic_dim,
            "hidden_size": self.hidden_size,
        }
        self.co_attn = CoAttention(hparams_coattn)

        enc_hparams = {
            'rnn_cell': {
                'type': 'LSTMCell',
                'kwargs': {
                    'num_units': self.hidden_size
                }
            }
        }

        default_hparams = UnidirectionalRNNEncoder.default_hparams()
        hparams_rnn = HParams(enc_hparams, default_hparams)

        self.lstm = UnidirectionalRNNEncoder(
            input_size=self.hidden_size,
            hparams=hparams_rnn.todict()
        )

        self.W_t_h = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size)

        self.W_t_ctx = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size)

        self.W_stop_s_1 = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size)

        self.W_stop_s = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size)

        self.W_stop = nn.Linear(
            in_features=self.hidden_size,
            out_features=2)

    def get_semantic_feature(self, tag_probs):
        batch_size = tag_probs.shape[0]
        normal_index = self.num_tags * tag_probs.new_ones(
            [batch_size, self.k])
        semantic_feature = self.embed(normal_index.to(torch.long))

        mask = tag_probs > self.threshold
        num_active = mask.to(torch.long).sum(1)
        mask = num_active > 0

        if mask.to(torch.long).sum() > 0:
            semantic_feature[mask] = self.embed(
                torch.topk(tag_probs[mask], self.k)[1])

        return semantic_feature

    def forward(self, v, a, prev_hidden):
        # TODO: Fill in return docstring
        r"""
        Return the visual_alignments, semantic_alignments for the loss function calculation
        Stack the visual_alignments, semantic_alignments at each time step of the sentence LSTM to
        obtain the alpha (visual_alignments) beta (semantic_alignments)
        Args:
            v (torch.Tensor): Visual features of image patches
            a (torch.Tensor): Semantic features. Word embeddings of predicted disease tags
            hidden (torch.Tensor): Previous hidden state in LSTM

        Returns:

        """
        context, visual_align, semantic_align = self.co_attn(
            v, a, prev_hidden[0])
        # Unsqueeze the second dimension to make it a 3-D tensor.
        # Same solution as in
        # https://github.com/ZexinYan/Medical-Report-Generation/blob/master/utils/models.py#L323

        _, hidden = self.lstm(
            context.unsqueeze(1),
            initial_state=prev_hidden)

        # _, hidden = self.lstm(context, state=prev_hidden)

        # Equation 5 in the paper
        topic = torch.tanh(
            self.W_t_h(hidden[0]) + self.W_t_ctx(context)
        )
        topic_var = torch.std(topic, dim=0).mean()

        # Equation 6 in the paper
        p_stop = torch.sigmoid(self.W_stop(
            torch.tanh(
                self.W_stop_s_1(prev_hidden[0]) + self.W_stop_s(hidden[0])
            )
        ))

        return hidden, topic, p_stop, visual_align, semantic_align, topic_var

    def init_hidden(self, batch_size):
        r"""Initialize hidden tensor

        Returns:
            Tuple[torch.Tensor]: Tuple of tensors with size
            [batch_size, hidden_size]

        """
        zeros = torch.zeros(batch_size, self.hidden_size)
        if torch.cuda.is_available():
            zeros = zeros.cuda()

        return (zeros, zeros)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            "hidden_size": 512,
            "visual_dim": 1024,
            "semantic_dim": 512,
            "num_tags": 13,
            "top_k_for_semantic": 3,
            "threshold": 0.5
        }

    @property
    def output_size(self):
        return self.lstm.output_size


class CustomTrainHelper(TrainingHelper):
    def __init__(
        self, *args,
        time_major=False,
        token_embedder=None,
        iters=None,
        k=200,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._time_major = time_major
        self.token_embedder = token_embedder
        self._inputs = None
        self._threshold = None
        self._iters = iters
        self._k = k

    def initialize(self, embedding_fn, inputs, sequence_length):
        if inputs is None:
            raise ValueError("`inputs` cannot be None for TrainingHelper")
        if sequence_length is None:
            raise ValueError(
                "`sequence_length` cannot be None for TrainingHelper")
        inputs: torch.Tensor
        sequence_length: torch.LongTensor

        if sequence_length.dim() != 1:
            raise ValueError(
                f"Expected 'sequence_length' to be a vector, "
                f"but received shape: {sequence_length.shape}")

        if not self._time_major:
            inputs = inputs.transpose(0, 1)  # make inputs time major
        times = torch.arange(
            sequence_length.max(), dtype=torch.long, device=inputs.device)
        times = times.unsqueeze(1).expand(-1, inputs.size(1))
        inputs = embedding_fn(inputs, times)

        self._inputs = inputs
        self._sequence_length = sequence_length
        self._zero_inputs = inputs.new_zeros(inputs[0].size())
        self._threshold = self._k / (self._k + np.exp(self._iters / self._k))
        self._threshold = np.max([self._threshold, 1.0])

        finished: torch.ByteTensor = (sequence_length == 0)
        all_finished = torch.all(finished).item()
        next_inputs = inputs[0] if not all_finished else self._zero_inputs

        return (finished, next_inputs)

    def next_inputs(self, embedding_fn,
                    time: int, outputs: torch.Tensor,
                    sample_ids: torch.LongTensor):
        del embedding_fn, outputs  # unused by next_inputs_fn
        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = torch.all(finished).item()

        rand = np.random.rand()
        if next_time < 2 or rand < self._threshold:
            next_inputs = self._inputs[next_time]
            next_inputs = (self._inputs[next_time] if not all_finished
                        else self._zero_inputs)
        else:
            embeddings = self.token_embedder(sample_ids)
            next_inputs = (embeddings if not all_finished
                        else self._zero_inputs)

        return (finished.cuda(), next_inputs)


class InferenceHelper(GreedyEmbeddingHelper):
    def __init__(
        self, *args,
        time_major=False,
        token_embedder=None,
        prefix_length=2,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._time_major = time_major
        self.token_embedder = token_embedder
        self._prefix_length = prefix_length
        self._inputs = None

    def initialize(self, embedding_fn, inputs, sequence_length):
        del embedding_fn, sequence_length
        if not self._time_major:
            inputs = inputs.transpose(0, 1)  # make inputs time major
        self._inputs = inputs
        finished = torch.zeros(self._start_tokens.shape[0], dtype=torch.bool)
        return (finished.cuda(), self._inputs[0])

    def next_inputs(self, embedding_fn,
                    time: int, outputs: torch.Tensor,
                    sample_ids: torch.LongTensor):
        del embedding_fn, outputs  # unused by next_inputs_fn
        next_time = time + 1
        if next_time < self._prefix_length:
            finished = torch.zeros(sample_ids.shape[0], dtype=torch.bool)
            next_inputs = self._inputs[next_time]
        else:
            finished = (sample_ids == self._end_token)
            all_finished = torch.all(finished).item()
            embeddings = self.token_embedder(sample_ids)
            next_inputs = (embeddings if not all_finished else self._inputs[1])
        # print(sample_ids)
        return (finished.cuda(), next_inputs)


class LstmWord(ModuleBase):
    """This is an implementation of 2nd in the hierarchy for 2 layered hierarchical LSTM
    implementation in Texar. In this particular application the first tier takes in the input
    which here is from co-attention and outputs a hidden state vector. At each time step of the
    tier-1 LSTM, the hidden state vector is fed into the tier 2 LSTM as a state tuple to output
    a sentence.

    NOTE: Run the LSTM sentence in a for loop range [0 max time steps] till termination
    Because at each time step we calculate the visual and semantic attention (stack at the end of
    the max timesteps to find the alpha and beta)
    Output the t needed for the Word lstm
    At each time step we produce a 0 or 1 to continue or stop

    This is run at every time step of Sentence LSTM as stated above
    It is Bernoulli variable
    p_pred shape [Batch size, 2]
    p_target shape [Batch size]

    Args:
        hparams (dict or HParams, optional): LstmWord hyperparameters. Missing
            hyperparameters will be set to default values.
            See :meth:`default_hparams` for the hyperparameter
            structure and default values.
        hidden_size (int): hidden_size the same as GNN output
        vocab_size (int):
        batch_size (int): batch size
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        self.hidden_size = self.hparams.hidden_size
        self.vocab_size = self.hparams.vocab_size

        # Embedding layer
        self.embedding = WordEmbedder(
            vocab_size=self.vocab_size,
            hparams={'dim': self.hidden_size}
        )

        enc_hparams = {
            'rnn_cell': {
                'type': 'LSTMCell',
                'kwargs': {
                    'num_units': self.hidden_size
                }
            }
        }

        default_hparams = BasicRNNDecoder.default_hparams()

        hparams_rnn = HParams(enc_hparams, default_hparams)

        self.decoder = BasicRNNDecoder(
            input_size=self.hidden_size,
            token_embedder=nn.Identity(),
            vocab_size=self.vocab_size,
            hparams=hparams_rnn.todict()
        )

    def forward(self, topic, train, inp=None, sentence_len=None, iters=None):
        """

        Args:
            topic (torch.Tensor): topic state from LstmSentence
            train (bool): If in training
            inp (torch.Tensor): Groundtruth tokens in a sentence.
            Only used in training
            sentence_len (int): Number of token in a sentence

        Returns:
            output (torch.Tensor): Generated output

        """
        # TODO: Make sure we could concatenate topics and input
        topic = topic.unsqueeze(1)
        batch_size = sentence_len.shape[0]
        start_tokens = torch.Tensor([START_TOKEN] * batch_size)
        start_tokens = start_tokens.long().cuda()
        end_tokens = END_TOKEN

        start_embeddings = self.embedding(start_tokens).unsqueeze(1)
        prefix_embeddings = torch.cat([topic, start_embeddings], dim=1)

        if train:
            helper = CustomTrainHelper(
                token_embedder=self.embedding, iters=iters)

            embeddings = self.embedding(inp)
            embeddings = torch.cat([prefix_embeddings, embeddings], dim=1)
            output, _, _ = self.decoder(
                helper=helper,
                inputs=embeddings,
                sequence_length=sentence_len)
        else:
            # Create helper
            helper = InferenceHelper(
                start_tokens=start_tokens,
                end_token=end_tokens,
                token_embedder=self.embedding
            )
            # Inference sample
            # here sentence length is the max_decoding_length
            output, _, _ = self.decoder(
                helper=helper,
                inputs=prefix_embeddings,
                max_decoding_length=30)

        return output

    def init_hidden(self, batch_size):
        r"""Initialize hidden tensor

        Returns:
            Tuple[torch.Tensor]: Tuple of tensors with size
            [batch_size, hidden_size]
        """
        zeros = torch.zeros(batch_size, self.hidden_size)
        if torch.cuda.is_available():
            zeros = zeros.cuda()

        return (zeros, zeros)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            "hidden_size": 512,
            "vocab_size": 512,
        }

    @property
    def output_size(self):
        r"""The feature size of forward output
        """
        return self.decoder.output_size
