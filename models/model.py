from typing import Dict, Any
import torch
import torch.nn as nn

from texar.torch import ModuleBase, HParams
from texar.torch.losses import sequence_sparse_softmax_cross_entropy
from texar.torch.evals import corpus_bleu
from texar.torch.utils import strip_special_tokens
from texar.torch.data.vocabulary import map_ids_to_strs
from texar.torch.data import Vocab

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

        self.ce_criterion = nn.CrossEntropyLoss()
        self.pathologies = self._hparams.pathologies
        self.vocab = Vocab(self._hparams.vocab_path)
        self.iters = 0

    def _init_model(self, model):
        if self.params:
            self.params += list(model.parameters())
        else:
            self.params = list(model.parameters())

        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def _init_mlc(self):
        model = MLC(self._hparams.sentence_lstm.num_tags)

        if self._hparams.mlc_weights:
            ckpt = torch.load(self._hparams.mlc_weights)
            m_state_dict = {}
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

        # Unpack the data batch
        visual_feature = batch['visual_feature'].cuda()
        token_tensor = batch['token_tensor'].cuda()
        stop_prob = batch['stop_prob'].cuda()

        # Generate semantic features from predicted tag probabilities
        tag_probs = self.mlc.get_tag_probs(visual_feature)
        active_tags = tag_probs > 0.5

        semantic_feature = self.sentence_lstm.get_semantic_feature(tag_probs)

        # Initialization
        stop_losses, word_losses, attention_losses = 0., 0., 0.
        sentence_states = self.sentence_lstm.init_hidden(batch_size)
        visual_align, semantic_align = [], []

        pred_words = []
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
            mask = teacher_words != self.vocab.pad_token_id
            sentence_len = mask.to(torch.long).sum(1)

            max_sentence_len = sentence_len.max()

            if max_sentence_len > 0:
                word_output = self.word_lstm(
                    topic, train=True,
                    inp=teacher_words, sentence_len=sentence_len + 1,
                    iters=self.iters)

                # per time step S of sentence LSTM
                word_losses += sequence_sparse_softmax_cross_entropy(
                    labels=teacher_words[:, :max_sentence_len],
                    logits=word_output.logits[:, 1:, :],
                    sequence_length=sentence_len)

                pred_words.append((word_output.sample_id, sentence_len))

        visual_align = torch.cat(visual_align, -1)
        semantic_align = torch.cat(semantic_align, -1)
        attention_losses = self.attn_loss(visual_align, semantic_align)

        # train_loss = self.lambda_stop * stop_losses \
        #                 + self.lambda_word * word_losses \
        #                 + self.lambda_attn * attention_losses
        train_loss = self.lambda_word * word_losses + \
            self.lambda_attn * attention_losses

        pred_paragrah = []
        for j in range(batch_size):
            paragrah = []
            for i in range(token_tensor.shape[1] - 1):
                sen = pred_words[i][0][j]
                len_sen = pred_words[i][1][j]
                sen = sen[1:len_sen + 1].cpu().tolist()
                if len(sen) > 0:
                    sen_tokens = self.vocab.map_ids_to_tokens_py(sen)
                    paragrah.append(' '.join(sen_tokens))

            paragrah = ' '.join(strip_special_tokens(paragrah))
            pred_paragrah.append(paragrah)

            # tag = active_tags[j]
            # indices = torch.nonzero(tag, as_tuple=True)
            # print(j)
            # print([self.pathologies[id] for id in indices[0]])
            # print(paragrah)

        target_paragrah = []
        for i in range(batch_size):
            paragrah = []
            for j in range(token_tensor.shape[1] - 1):
                sen = token_tensor[i, j].long()
                mask = sen != self.vocab.pad_token_id
                len_sen = mask.to(torch.long).sum()
                if len_sen > 0:
                    sen = sen[:len_sen].cpu().tolist()
                    sen_tokens = self.vocab.map_ids_to_tokens_py(sen)
                    paragrah.append(' '.join(sen_tokens))

            paragrah = ' '.join(strip_special_tokens(paragrah))
            target_paragrah.append([paragrah])

        bleu, bleu_1, bleu_2, bleu_3, bleu_4 = corpus_bleu(
            target_paragrah, pred_paragrah, return_all=True)

        return {
            "loss": train_loss,
            "stop_loss": stop_losses,
            "word_loss": word_losses,
            "attention_loss": attention_losses,
            "topic_var": topic_var,
            "bleu": bleu,
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
        }

    def predict(self, batch):
        self.iters += 1
        batch_size = batch.batch_size

        # Unpack the data batch
        visual_feature = batch['visual_feature'].cuda()
        token_tensor = batch['token_tensor'].cuda()

        # Generate semantic features from predicted tag probabilities
        tag_probs = self.mlc.get_tag_probs(visual_feature)
        active_tags = tag_probs > 0.5
        semantic_feature = self.sentence_lstm.get_semantic_feature(tag_probs)

        # Initialization
        sentence_states = self.sentence_lstm.init_hidden(batch_size)
        stopped_mask = torch.zeros(batch_size).to(torch.bool)

        stopped_mask = torch.zeros([batch_size], dtype=torch.bool)
        pred_words = []
        max_sentence_num = 0
        count = 0
        while True:
            # Obtain the topic and pre_stop from the sentence LSTM
            sentence_states, topic, pred_stop, _, _, _ = self.sentence_lstm(
                    visual_feature,
                    semantic_feature,
                    sentence_states)

            stopped = torch.nonzero(pred_stop[:, 0] > 0.5)
            stopped_mask[stopped] = True
            if torch.all(stopped_mask):
                break

            word_output = self.word_lstm(
                topic, train=False,
                iters=self.iters)
            max_sentence_num += 1

            word_output.sample_id[stopped_mask] = self.vocab.eos_token_id
            pred_words.append(word_output.sample_id)
            count += 1
            if count > 5:
                exit()

        pred_paragrah = []
        for j in range(batch_size):
            paragrah = []
            for i in range(max_sentence_num):
                sen = pred_words[i][j]
                mask = sen == self.vocab.eos_token_id
                first_eos_index = mask.nonzero(as_tuple=True)[0][0]
                if first_eos_index > 0:
                    sen = sen[1:first_eos_index+1].cpu().tolist()
                    sen_tokens = self.vocab.map_ids_to_tokens_py(sen)
                    paragrah.append(' '.join(sen_tokens))

            # paragrah = ' '.join(strip_special_tokens(paragrah))
            paragrah = ' '.join(paragrah)
            pred_paragrah.append(paragrah)
            print(j, paragrah)

            # tag = active_tags[j]
            # indices = torch.nonzero(tag, as_tuple=True)
            # print(j)
            # print([self.pathologies[id] for id in indices[0]])
            # print(paragrah)

        target_paragrah = []
        for i in range(batch_size):
            paragrah = []
            for j in range(token_tensor.shape[1] - 1):
                sen = token_tensor[i, j].long()
                mask = sen != self.vocab.pad_token_id
                len_sen = mask.to(torch.long).sum()
                if len_sen > 0:
                    sen = sen[:len_sen].cpu().tolist()
                    sen_tokens = self.vocab.map_ids_to_tokens_py(sen)
                    paragrah.append(' '.join(sen_tokens))

            paragrah = ' '.join(strip_special_tokens(paragrah))
            target_paragrah.append([paragrah])

        bleu, bleu_1, bleu_2, bleu_3, bleu_4 = corpus_bleu(
            target_paragrah, pred_paragrah, return_all=True)

        return {
            'bleu': bleu,
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4,
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
                    "num_tags": 210,
                    "top_k_for_semantic": 10,
                    "threshold": 0.5
                },
                "word_lstm":{
                    "hidden_size": 512,
                    "vocab_size": 1004,
                    "BOS": 2,
                    "EOS": 3,
                },
                "lambda_stop": 1.,
                "lambda_word": 1.,
                "lambda_attn": 1.,
                "mlc_weights": None,
                "pathologies": None,
                "vocab_path": None,
        }
