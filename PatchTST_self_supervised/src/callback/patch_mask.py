import logging
import torch
from torch import nn

from .core import Callback

# Cell
class PatchCB(Callback):

    def __init__(self, patch_len, stride ):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        # learner get the transformed input
        self.learner.xb = xb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len]           


class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio, mask_value,
                        mask_when_pred:bool=False):
        """
        Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
        Args:
            patch_len:        patch length
            stride:           stride
            mask_ratio:       mask ratio
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    def before_fit(self):
        # overwrite the predefined loss function
        self.learner.loss_func = self._loss        
        device = self.learner.device       
 
    def before_forward(self): self.patch_masking()
        
    def patch_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
        xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio, self.mask_value)   # xb_mask: [bs x num_patch x n_vars x patch_len]
        #print("Mean mask incidence:", self.mask.mean())
        self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
        self.learner.xb = xb_mask       # learner.xb: masked 4D tensor    
        self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor

    def _loss(self, preds, target):        
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """
        #print("PatchMaskCB Loss, pred and target shapes", preds.shape, target.shape)
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask).sum() / self.mask.sum()
        return loss


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


class Patch(nn.Module):
    # TODO Does not seem to be used anywhere
    def __init__(self,seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len  + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)                 # xb: [bs x num_patch x n_vars x patch_len]
        return x



class ObservationMaskCB(Callback):
    """
    Function that mask observations in the input, before patching
    
    Function that evaluates loss on the masked items only
    """

    def __init__(self, mask_ratio, mask_value):
        """
        Mask random observations in the raw (prepatched) input data"""
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    def before_fit(self):
        # overwrite the predefined loss function
        self.learner.loss_func = self._loss
        device = self.learner.device

    def before_forward(self):
        self.masking()

    def masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x seq_len x n_vars]
        """
        xb_mask, _, mask, _ = random_masking_3D(self.xb, self.mask_ratio, self.mask_value)   # xb_mask: [bs x seq_len x n_vars]
        #print("Mean mask incidence:", self.mask.mean())
        self.mask = mask.bool()    # mask: [seq_len x n_vars]
        self.xb = xb_mask      # learner.xb: masked 3D input tensor
        self.learner.yb = self.xb      # Raw input as expected output

    def _loss(self, preds, target):
        """
        preds:   [bs x seq_len x n_vars]
        targets: [bs x seq_len x n_vars]
        """
        loss = (preds - target) ** 2
        loss = loss.mean(dim=0) # Mean across batch
        loss = (loss * self.mask).sum() / self.mask.sum() # Mean of means with mask
        logging.debug(f"ObservationMaskCB custom loss: preds {preds.shape}, target {target.shape}, mask {self.mask.shape}")
        return loss

def random_masking(xb, mask_ratio, mask_value=0):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, nvars,device=xb.device)  # noise in [0, 1], bs x L x nvars
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]                                              # ids_keep: [bs x len_keep x nvars]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len]
   
    # removed x
    x_removed = torch.ones(bs, L-len_keep, nvars, D, device=xb.device) * mask_value  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def random_masking_3D(xb, mask_ratio, mask_value=0):
    # xb: [bs x seq_len x n_vars]
    """
    Args:
        xb: [bs x seq_len x n_vars]
        mask_ratio: float

    """
    xb = xb.transpose(0, 2).clone() # [n_vars x seq_len x bs]
    n_vars, seq_len, bs = xb.shape
    x = xb.clone()
    
    len_keep = int(seq_len * (1 - mask_ratio))
        
    noise = torch.rand(n_vars, seq_len, device=xb.device)  # noise in [0, 1], n_vars x seq_len
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [n_vars x seq_len]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]                                                 # ids_keep: [n_vars x len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, bs ))        # x_kept: [n_vars x len_keep x bs]
   
    # removed x
    x_removed = torch.ones(n_vars, seq_len-len_keep, bs, device=xb.device) * mask_value   # x_removed: [n_vars x (seq_len-len_keep) x bs]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x seq_len x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,bs ))    # x_masked: [???]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([n_vars, seq_len], device=x.device)                                          # mask: [n_vars x seq_len]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [n_vars x seq_len]

    x_masked = x_masked.transpose(0, 2) # [bs x seq_len x n_vars]
    mask = mask.transpose(0,1)      # [seq_len x n_var]
    return x_masked, x_kept, mask, ids_restore


if __name__ == "__main__":
    bs, L, nvars, D = 2,20,4,5
    xb = torch.randn(bs, L, nvars, D)
    xb_mask, mask, ids_restore = create_mask(xb, mask_ratio=0.5)
    breakpoint()


