from PatchTST_self_supervised.src.callback.patch_mask import ObservationMaskCB, random_masking_3D
import torch


def test_observation_mask_cb(batch):

    MASK_RATIO = .4
    torch.manual_seed(22219)
    xb_mask, _, mask, _ = random_masking_3D(batch, MASK_RATIO)
    mask =  mask.bool()

    assert (batch.size()==xb_mask.size()) # Masked output same size of input
    assert all(batch[0][mask==False] == xb_mask[0][mask==False]) # Non masked entries match input
    assert xb_mask[0][mask==True].sum()==0 # Masked entires are all zero.
