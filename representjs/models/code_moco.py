import torch
from torch import nn
import horovod.torch as hvd

from models.encoder import CodeEncoder, CodeEncoderLSTM


class MoCoTemplate(nn.Module):
    """From https://github.com/facebookresearch/moco/blob/master/moco/builder.py"""

    def __init__(self, d_rep=128, K=61440, m=0.999, T=0.07, use_horovod=False, encoder_params={}):  # 61440 = 2^12 * 3 * 5
        """
        d_rep: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        self.config = dict(**{"moco_num_keys": K, "moco_momentum": m, "moco_temperature": T}, **encoder_params)
        self.K = K
        self.m = m
        self.T = T
        self.use_horovod = use_horovod

        self.encoder_q = self.make_encoder(**encoder_params)
        self.encoder_k = self.make_encoder(**encoder_params)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        # for param_k in self.encoder_k.parameters():
        #     param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(d_rep, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def make_encoder(self, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.use_horovod:
            keys = hvd.allgather(keys)
        else:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        # ptr = int(self.queue_ptr)
        ptr = int(self.queue_ptr.item())
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue = torch.cat([self.queue[:, :ptr], keys.T, self.queue[:, ptr + batch_size :]], dim=1).detach()
        # self.queue = self.queue.clone()
        # self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def embed_x(self, img, lens):
        return self.encoder_q(img, lens)

    def forward(self, im_q, im_k, lengths_k, lengths_q, q=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            lengths_k: sequence length, [B,]
            lengths_q: sequence length, [B,]
            q: queries, pre-computed embedding of im_q, [N, C]
        Output:
            logits, targets
        """

        # compute query features
        if q is None:
            q = self.encoder_q(im_q, lengths_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k, lengths_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", *[q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", *[q, self.queue.detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        # print("world size", torch.distributed.get_world_size())
        self._dequeue_and_enqueue(k)

        return logits, labels


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class CodeMoCo(MoCoTemplate):
    def __init__(self, n_tokens, d_model=512, d_rep=128, K=107520, m=0.999, T=0.07, use_horovod=False, encoder_config={}, pad_id=None):
        super().__init__(
            d_rep,
            K,
            m,
            T,
            use_horovod=use_horovod,
            encoder_params=dict(n_tokens=n_tokens, d_model=d_model, d_rep=d_rep, pad_id=pad_id, **encoder_config),
        )

    def make_encoder(
        self,
        n_tokens,
        d_model,
        d_rep,
        pad_id=None,
        encoder_type="transformer",
        lstm_project_mode="hidden",
        n_encoder_layers=6,
        dropout=0.1,
        **kwargs
    ):
        if encoder_type == "transformer":
            return CodeEncoder(
                n_tokens, project=True, pad_id=pad_id, d_model=d_model, d_rep=d_rep, n_encoder_layers=n_encoder_layers, **kwargs
            )
        elif encoder_type == "lstm":
            return CodeEncoderLSTM(
                n_tokens=n_tokens,
                d_model=d_model,
                d_rep=d_rep,
                n_encoder_layers=n_encoder_layers,
                dropout=dropout,
                pad_id=pad_id,
                project=lstm_project_mode,
            )
        else:
            raise ValueError

    def forward(self, im_q, im_k, lengths_q, lengths_k, q=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        return super().forward(im_q, im_k, lengths_q, lengths_k, q=q)
