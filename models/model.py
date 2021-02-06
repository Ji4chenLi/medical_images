from typing import Dict, Any
import torch
import torch.nn as nn

from texar.torch import ModuleBase, HParams
from texar.torch.losses import sequence_sparse_softmax_cross_entropy

from models.cv_model import MLC
from models.nlp_model import LstmSentence, LstmWord


class MedicalReportGenerator(ModuleBase):

    def __init__(self, hparams):
        super().__init__()
        self._hparams = HParams(hparams, self.default_hparams())
        self.params = None

        self.mlc = self._init_mlc()
        self.sentence_lstm = self._init_sentence_lstm()
        self.word_lstm = self._init_word_model()

        self.lambda_stop = self._hparams.lambda_stop
        self.lambda_word = self._hparams.lambda_word
        self.lambda_attn = self._hparams.lambda_attn

        self.ce_criterion = nn.CrossEntropyLoss(
            size_average=False, reduce=False)

    def _init_model(self, model):
        if self.params:
            self.params += list(model.parameters())
        else:
            self.params = list(model.parameters())

        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def _init_mlc(self):
        model = MLC()
        if self._hparams.mlc_weights:
            ckpt = torch.load(self._hparams.mlc_weights)
            m_state_dict = model.state_dict()
            len_mlc = len('mlc.')

            for k in ckpt.model.keys():
                new_k = k[len_mlc:]
                m_state_dict[new_k] = ckpt.model[k]

        model = self._init_model(model)

        return model

    def _init_sentence_lstm(self):
        model = LstmSentence(self._hparams.sentence_lstm)
        model = self._init_model(model)

        return model

    def _init_word_model(self):
        model = LstmWord(self._hparams.word_lstm)
        model = self._init_model(model)

        return model

    def attn_loss(self, alpha, beta):
        r"""Attention loss function. Weigh it in the main loop as per lambda

        Args:
            alpha (): Size: [batch_size, N, S].
            N is the number of visual features.
            S is the number of time steps in Sentence LSTM.

            beta (): Size: [batch_size, M, S].
            M is the number of semantic features.
            S is the number of time steps in Sentence LSTM

        Returns:
            loss (torch.Tensor): Calculated attention loss
        """

        visual_attn_loss = torch.sum(
            (1 - torch.sum(alpha, -1)) ** 2, -1)

        semantic_attn_loss = torch.sum(
            (1 - torch.sum(beta, -1)) ** 2, -1)

        return torch.mean(visual_attn_loss + semantic_attn_loss)

    def forward(self, batch):
        batch_size = batch.batch_size

        visual_feature = batch['visual_feature'].cuda()
        token_tensor = batch['token_tensor'].cuda()
        stop_prob = batch['stop_prob'].cuda()

        stop_losses, word_losses = 0., 0.

        tag_probs = self.mlc.get_tag_probs(visual_feature)
        semantic_feature = self.sentence_lstm.get_semantic_feature(tag_probs)

        sentence_states = self.sentence_lstm.init_hidden(batch_size)
        prev_hidden_states = self.sentence_lstm.init_hidden(batch_size)

        visual_align, semantic_align = [], []

        for sentence_index in range(token_tensor.shape[1]):
            _, hidden_states, topic, pred_stop, \
                v_align, s_align = self.sentence_lstm(
                    visual_feature, semantic_feature,
                    sentence_states, prev_hidden_states)

            prev_hidden_states = sentence_states
            sentence_states = hidden_states

            visual_align.append(v_align.unsqueeze(-1))
            semantic_align.append(s_align.unsqueeze(-1))

            stop_losses += self.ce_criterion(
                pred_stop,
                stop_prob[:, sentence_index].to(torch.long)
            ).sum()

            # Preclude the <start> token
            teacher_words = token_tensor[:, sentence_index, 1:].long()
            sentence_len = (teacher_words != 0).to(torch.long).sum(1)

            max_sentence_len = sentence_len.max()

            if max_sentence_len > 0:
                # TODO: Make sure the sentence_len is set correctly
                word_output = self.word_lstm(
                    topic, train=True,
                    inp=teacher_words, sentence_len=sentence_len)

                # per time step S of sentence LSTM
                word_losses += sequence_sparse_softmax_cross_entropy(
                    labels=teacher_words[:, :max_sentence_len],
                    logits=word_output.logits,
                    sequence_length=sentence_len)

        visual_align = torch.cat(visual_align, -1)
        semantic_align = torch.cat(semantic_align, -1)
        attention_losses = self.attn_loss(visual_align, semantic_align)
        batch_loss = self.lambda_stop * stop_losses \
                        + self.lambda_word * word_losses \
                        + self.lambda_attn * attention_losses

        return batch_loss

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
                "sentence_lstm": {
                    "hidden_size": 512,
                    "num_units": 512,
                    "visual_dim": 1024,
                    "semantic_dim": 512,
                    "num_tags": 14,
                    "top_k_for_semantic": 3,
                },
                "word_lstm":{
                    "hidden_size": 512,
                    "vocab_size": 12432,
                },
                "lambda_stop": 1.,
                "lambda_word": 1.,
                "lambda_attn": 1.,
                "mlc_weights": None
        }


# for word_index in range(1, token_tensor.shape[2]):
#     teacher_words = token_tensor[:, sentence_index, :word_index]
#     teacher_words = teacher_words.long()
#     pred_words = self.word_lstm.forward(topic, train=True, inp=teacher_words)

#     word_mask = (teacher_words > 0).float()
#     target_word = token_tensor[:, sentence_index, word_index]
#     word_losses += torch.sum(
#         self.ce_criterion(pred_words, target_word) * word_mask)