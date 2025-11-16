import os
import pdb
import time
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class SimpleCTCBeamDecoder:
    """
    Lightweight pure PyTorch beam-search decoder so that we do not depend on
    the external ctcdecode package. It implements a Viterbi-style search that
    keeps the top-N prefixes at every timestep and merges identical prefixes
    by keeping the best score. The interface is aligned with ctcdecode so the
    rest of the pipeline can stay unchanged.
    """

    def __init__(self, blank_id=0, beam_width=10):
        self.blank_id = blank_id
        self.beam_width = beam_width

    def _update(self, beam_dict, prefix, score):
        if prefix in beam_dict:
            if score > beam_dict[prefix]:
                beam_dict[prefix] = score
        else:
            beam_dict[prefix] = score

    def _beam_search_single(self, log_probs):
        """
        Args:
            log_probs (Tensor): (T, C) log probabilities for a single sample.
        Returns:
            List[Tuple[Tuple[int, ...], float]]: best prefixes and scores.
        """
        Timesteps, _ = log_probs.shape
        beam = {tuple(): 0.0}

        for t in range(Timesteps):
            next_beam = {}
            log_probs_t = log_probs[t]
            blank_score = float(log_probs_t[self.blank_id])
            for prefix, score in beam.items():
                # Stay on the current prefix via blank
                self._update(next_beam, prefix, score + blank_score)

                topk = min(self.beam_width, log_probs_t.numel())
                values, indices = torch.topk(log_probs_t, k=topk)
                for value, idx in zip(values.tolist(), indices.tolist()):
                    if idx == self.blank_id:
                        continue
                    new_prefix = prefix + (int(idx),)
                    self._update(next_beam, new_prefix, score + float(value))

            if not next_beam:
                next_beam = beam

            sorted_beam = sorted(next_beam.items(), key=lambda x: x[1], reverse=True)
            beam = dict(sorted_beam[:self.beam_width])

        # Ensure we always return beam_width beams, padded with blanks if needed
        beam_items = sorted(beam.items(), key=lambda x: x[1], reverse=True)
        if not beam_items:
            beam_items = [(tuple(), float("-inf"))]
        while len(beam_items) < self.beam_width:
            beam_items.append((tuple(), float("-inf")))
        return beam_items[:self.beam_width]

    def decode(self, probs, seq_lgt):
        batch, max_timesteps, num_classes = probs.shape
        seq_lgt = seq_lgt.cpu().tolist()
        probs = probs.cpu()

        all_sequences = []
        all_scores = []
        max_output_length = 0

        for batch_idx in range(batch):
            tgt_len = seq_lgt[batch_idx]
            sample_probs = probs[batch_idx, :tgt_len]
            if sample_probs.numel() == 0:
                beam_items = [(tuple(), float("-inf")) for _ in range(self.beam_width)]
            else:
                log_probs = torch.log(torch.clamp(sample_probs, min=1e-12))
                beam_items = self._beam_search_single(log_probs)

            sequences = [list(prefix) for prefix, _ in beam_items]
            scores = [score for _, score in beam_items]
            max_output_length = max(max_output_length, max((len(seq) for seq in sequences), default=0))

            all_sequences.append(sequences)
            all_scores.append(scores)

        if max_output_length == 0:
            max_output_length = 1

        beam_result = torch.full((batch, self.beam_width, max_output_length),
                                 fill_value=self.blank_id, dtype=torch.int32)
        beam_scores = torch.full((batch, self.beam_width), fill_value=float("inf"))
        out_seq_len = torch.zeros((batch, self.beam_width), dtype=torch.int32)
        timesteps = torch.zeros((batch, self.beam_width, max_output_length), dtype=torch.int32)

        for batch_idx, (sequences, scores) in enumerate(zip(all_sequences, all_scores)):
            for beam_idx, (seq, score) in enumerate(zip(sequences, scores)):
                length = len(seq)
                if length > 0:
                    beam_result[batch_idx, beam_idx, :length] = torch.tensor(seq, dtype=torch.int32)
                out_seq_len[batch_idx, beam_idx] = length
                beam_scores[batch_idx, beam_idx] = -score  # positive scores like ctcdecode
                if length > 0:
                    timesteps[batch_idx, beam_idx, :length] = torch.arange(length, dtype=torch.int32)

        return beam_result, beam_scores, timesteps, out_seq_len


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        self.ctc_decoder = SimpleCTCBeamDecoder(blank_id=blank_id, beam_width=10)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list
