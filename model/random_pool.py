import torch
import torch.nn as nn


class RandomPoolDAN(nn.Module):
    """
    Random pooling for the dense DAN tensor format.

    Replaces the learned SAGC retention score with a uniform random score,
    so which operations survive pooling is decided by chance instead of a
    learned projection. Protection/exclusion rules, k, and the masking
    logic that rebuilds op_mask are unchanged.

    Differences to the SAGC version, caused by the DAN data format:
      * DAN has no explicit operation adjacency. The OAB uses a roll trick
        on the dense tensor, so compacting the kept nodes in their original
        order reconnects the chain automatically (o1-o2-o3 with o2 removed
        becomes o1-o3). Only op_mask has to be rebuilt.
      * There is no padding in the same_op_nums setting, so the pad mask
        reduces to the deleted mask. In the various_op_nums setting the
        caller folds padding nodes into the deleted mask.

    Protection and exclusion rules (identical to the SAGC version):
      * eligible operations get score +inf and are always kept
      * deleted (completed) operations get score -inf
      * k = max(num_protected, k_target), capped at N
      * if fewer non-deleted operations than k exist, deleted operations
        fill the remaining slots. Their embeddings are zeroed and they are
        disconnected in the rebuilt op_mask, so they act as neutral padding
        and keep k uniform across the batch.
    """

    def __init__(self, ratio, k_mode="ops"):
        """
        :param ratio:   pooling ratio, interpretation depends on k_mode
        :param k_mode:  'ops'  -> k_target = ratio * N
                        'jobs' -> k_target = ratio * num_jobs
        """
        super().__init__()
        self.ratio = ratio
        self.k_mode = k_mode
        self.diagnostic_mode = False
        self._last_diag = None

    @staticmethod
    def _rebuild_op_mask(kept_jobs, kept_deleted):
        """
        Rebuild the [B, k, 3] op_mask for the pooled tensor.

        Column 0 masks the predecessor, column 2 the successor, column 1
        (self) is never masked, exactly as in DAN. Position i attends to
        position i-1 only if both belong to the same job and neither is a
        deleted filler node. Same for i+1.
        """
        B, k = kept_jobs.shape
        device = kept_jobs.device
        op_mask = torch.zeros(B, k, 3, dtype=torch.float32, device=device)

        # predecessor side
        op_mask[:, 0, 0] = 1
        if k > 1:
            job_change = (kept_jobs[:, 1:] != kept_jobs[:, :-1])
            cut_pre = job_change | kept_deleted[:, 1:] | kept_deleted[:, :-1]
            op_mask[:, 1:, 0] = torch.maximum(op_mask[:, 1:, 0], cut_pre.float())

        # successor side
        op_mask[:, -1, 2] = 1
        if k > 1:
            cut_sub = job_change | kept_deleted[:, :-1] | kept_deleted[:, 1:]
            op_mask[:, :-1, 2] = torch.maximum(op_mask[:, :-1, 2], cut_sub.float())

        return op_mask

    def compute_k(self, N, num_jobs, num_protected):
        if self.k_mode == "jobs":
            k_target = max(1, int(self.ratio * num_jobs))
        elif self.k_mode == "ops":
            k_target = max(1, int(self.ratio * N))
        else:
            raise ValueError(f"Unknown k_mode '{self.k_mode}'. Use 'jobs' or 'ops'.")
        k = max(num_protected, k_target)
        k = min(k, N)
        return k

    def forward(self, h, candidate, opes_appertain, eligible_opes, deleted_opes):
        """
        :param h:              operation embeddings [B, N, d]
        :param candidate:      candidate operation indices [B, J]
        :param opes_appertain: job index per operation [B, N]
        :param eligible_opes:  bool [B, N], True = must be kept
        :param deleted_opes:   bool [B, N], True = completed or padding,
                               excluded from pooling
        :return:
            h_pooled          [B, k, d]
            op_mask_pooled    [B, k, 3]
            candidate_pooled  [B, J]  candidate indices remapped to pooled
                              positions (fallback 0 for candidates that are
                              not in the pooled graph, harmless because their
                              comp_idx entries are zero and their actions are
                              masked by dynamic_pair_mask)
            top_idx           [B, k]  pooled position -> original position
        """
        B, N, d = h.shape
        device = h.device

        # 1) protection and exclusion, scored purely at random
        protect_mask = eligible_opes & ~deleted_opes
        sel_scores = torch.rand(B, N, device=device)
        sel_scores = sel_scores.masked_fill(deleted_opes, float('-inf'))
        sel_scores = sel_scores.masked_fill(protect_mask, float('inf'))

        num_protected = int(protect_mask.sum(dim=-1).max().item())
        k = self.compute_k(N, int(opes_appertain.max().item()) + 1, num_protected)

        # 2) select and sort, so the compacted tensor keeps the original order
        top_idx = torch.topk(sel_scores, k, dim=-1).indices
        top_idx, _ = torch.sort(top_idx, dim=-1)

        # 3) gather embeddings for the kept positions
        h_pooled = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, d))

        # 4) deleted filler nodes: zero their embeddings so they behave like
        #    DAN's own deleted nodes (excluded by nonzero_averaging and inert
        #    in the next attention layer)
        kept_deleted = deleted_opes.gather(1, top_idx)
        h_pooled = h_pooled.masked_fill(kept_deleted.unsqueeze(-1), 0.0)

        # 5) rebuild op_mask on the pooled tensor
        kept_jobs = opes_appertain.gather(1, top_idx)
        op_mask_pooled = self._rebuild_op_mask(kept_jobs, kept_deleted)

        # 6) remap candidate indices to pooled positions
        reverse_map = torch.zeros(B, N, dtype=torch.long, device=device)
        pooled_positions = torch.arange(k, device=device).unsqueeze(0).expand(B, -1)
        reverse_map.scatter_(1, top_idx, pooled_positions)
        candidate_pooled = reverse_map.gather(1, candidate.long())

        if self.diagnostic_mode:
            self._last_diag = {
                "sel_scores": sel_scores.detach().cpu(),
                "top_idx": top_idx.detach().cpu(),
                "k": k,
            }

        return h_pooled, op_mask_pooled, candidate_pooled, top_idx
