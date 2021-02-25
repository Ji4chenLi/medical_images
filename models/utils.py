import torch
import numpy as np
from texar.torch.modules.decoders.decoder_helpers \
    import GreedyEmbeddingHelper, TrainingHelper


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
        if inputs is None:
            raise ValueError("`inputs` cannot be None for InferenceHelper")
        inputs: torch.Tensor

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

        return (finished.cuda(), next_inputs)
