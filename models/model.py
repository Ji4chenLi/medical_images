from typing import Dict, Any
import pickle
import torch
import torch.nn as nn

from texar.torch import ModuleBase, HParams
from texar.torch.losses import sequence_sparse_softmax_cross_entropy

from models.cv_model import MLC
from models.nlp_model import LstmSentence, LstmWord

from build_vocab import PAD_TOKEN, Vocabulary


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

        self.ce_criterion = nn.CrossEntropyLoss()
        self.iters = 0
        self.vocab = pickle.load(
            open('../jiachen_medical_images/preprocessed/vocab.pkl', 'rb'))

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

        model.load_state_dict(m_state_dict)

        for p in model.parameters():
            p.requires_grad = False

        if torch.cuda.is_available():
            model = model.cuda()

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
            ((1 - torch.sum(alpha, -1)) ** 2), -1)

        semantic_attn_loss = torch.sum(
            ((1 - torch.sum(beta, -1)) ** 2), -1)

        return torch.mean(visual_attn_loss + semantic_attn_loss)

    def forward(self, batch):
        self.iters += 1
        batch_size = batch.batch_size
        # batch_size = 1

        # # Unpack the data batch
        # visual_feature = batch['visual_feature'][0].unsqueeze(0).cuda()
        # token_tensor = batch['token_tensor'][0].unsqueeze(0).cuda()
        # stop_prob = batch['stop_prob'][0].unsqueeze(0).cuda()

        # Unpack the data batch
        visual_feature = batch['visual_feature'].cuda()
        token_tensor = batch['token_tensor'].cuda()
        stop_prob = batch['stop_prob'].cuda()


        # Generate semantic features from predicted tag probabilities
        tag_probs = self.mlc.get_tag_probs(visual_feature)
        semantic_feature = self.sentence_lstm.get_semantic_feature(tag_probs)

        # Initialization
        stop_losses, word_losses, attention_losses = 0., 0., 0.
        sentence_states = self.sentence_lstm.init_hidden(batch_size)
        visual_align, semantic_align = [], []

        for sentence_index in range(token_tensor.shape[1]):
            # Obtain the topic and pre_stop from the sentence LSTM
            sentence_states, topic, pred_stop, \
                 v_align, s_align, topic_var = self.sentence_lstm(
                    visual_feature,
                    semantic_feature,
                    sentence_states)

            visual_align.append(v_align.unsqueeze(-1))
            semantic_align.append(s_align.unsqueeze(-1))

            stop_losses += self.ce_criterion(
                pred_stop,
                stop_prob[:, sentence_index].to(torch.long)
            ).sum()

            # Note the 1 below is to preclude the <start> token
            teacher_words = token_tensor[:, sentence_index, :].long()
            # Mask out all PAD_TOKEN
            mask = teacher_words != PAD_TOKEN
            sentence_len = mask.to(torch.long).sum(1)

            max_sentence_len = sentence_len.max()

            if max_sentence_len > 0:
                word_output = self.word_lstm(
                    topic, train=True,
                    inp=teacher_words, sentence_len=sentence_len + 1,
                    iters=self.iters)

                # for i, sentence in enumerate(word_output.sample_id):
                #     s = []
                #     t = []
                #     for j, word in enumerate(sentence[1:]):
                #         if j < sentence_len[i]:
                #             s.append(
                #                 self.vocab.get_word_by_id(word.cpu().tolist()))
                #             t.append(
                #                 self.vocab.get_word_by_id(teacher_words[i, j].cpu().tolist()))
                #     print(' '.join(s))
                #     print(' '.join(t))
                #     print(i)
                # exit()

                # per time step S of sentence LSTM
                word_losses += sequence_sparse_softmax_cross_entropy(
                    labels=teacher_words[:, :max_sentence_len],
                    logits=word_output.logits[:, 1:, :],
                    sequence_length=sentence_len)

        visual_align = torch.cat(visual_align, -1)
        semantic_align = torch.cat(semantic_align, -1)
        attention_losses = self.attn_loss(visual_align, semantic_align)

        train_loss = self.lambda_stop * stop_losses \
                        + self.lambda_word * word_losses \
                        + self.lambda_attn * attention_losses

        return {
            "loss": train_loss,
            "stop_loss": stop_losses,
            "word_loss": word_losses,
            "attention_loss": attention_losses,
            "topic_var": topic_var
        }

    def predict(self, batch):
        self.iters += 1
        batch_size = batch.batch_size
        # batch_size = 1

        # # Unpack the data batch
        # visual_feature = batch['visual_feature'][0].unsqueeze(0).cuda()
        # token_tensor = batch['token_tensor'][0].unsqueeze(0).cuda()

        # Unpack the data batch
        visual_feature = batch['visual_feature'].cuda()
        token_tensor = batch['token_tensor'].cuda()

        # Generate semantic features from predicted tag probabilities
        tag_probs = self.mlc.get_tag_probs(visual_feature)
        semantic_feature = self.sentence_lstm.get_semantic_feature(tag_probs)

        # Initialization
        sentence_states = self.sentence_lstm.init_hidden(batch_size)
        stopped_mask = torch.zeros(batch_size).to(torch.bool)

        for sentence_index in range(token_tensor.shape[1]):
            # Obtain the topic and pre_stop from the sentence LSTM
            sentence_states, topic, pred_stop, _, _, _ = self.sentence_lstm(
                    visual_feature,
                    semantic_feature,
                    sentence_states)
            stopped = torch.nonzero(pred_stop[:, 0] > 0.5).squeeze().tolist()
            stopped_mask[stopped] = True
            # print(stopped_mask)
            # Note the 1 below is to preclude the <start> token
            teacher_words = token_tensor[:, sentence_index, :].long()
            # print(teacher_words)
            # Mask out all PAD_TOKEN
            mask = teacher_words != PAD_TOKEN
            sentence_len = mask.to(torch.long).sum(1)
            sentence_len[stopped_mask] = 0

            max_sentence_len = sentence_len.max()
            if max_sentence_len > 0:
                word_output = self.word_lstm(
                    topic, train=False,
                    sentence_len=sentence_len,
                    iters=self.iters)

                for i, sentence in enumerate(word_output.sample_id):
                    s = []
                    t = []
                    for j, word in enumerate(sentence[1:]):
                        s.append(self.vocab.get_word_by_id(word.cpu().tolist()))
                        if j < sentence_len[i]:
                            t.append(
                                self.vocab.get_word_by_id(teacher_words[i, j].cpu().tolist()))
                    print(' '.join(s))
                    print(' '.join(t))
                    print(i)
                    
                exit()

                # # per time step S of sentence LSTM
                # word_losses += sequence_sparse_softmax_cross_entropy(
                #     labels=teacher_words[:, :max_sentence_len],
                #     logits=word_output.logits,
                #     sequence_length=sentence_len)

        word_losses /= batch_size
        return {
        }

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
                "sentence_lstm": {
                    "hidden_size": 512,
                    "visual_dim": 1024,
                    "semantic_dim": 512,
                    "num_tags": 13,
                    "top_k_for_semantic": 3,
                    "threshold": 0.5
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