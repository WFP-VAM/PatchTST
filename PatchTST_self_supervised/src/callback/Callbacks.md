# Callbacks


## SetupLearnerCB
- Moves model to device before fitting
- Moves data to device before each batch train/validation/test step.
- Decide is GPU if available else CPU


## RevInCB
- Normalizes input before forward pass
- Denormalizes predictions after forward pass


## ObservationMaskCB
- Added by WFP APP-GRS
- Applies a mask over the `seq_len` x `n_vars` dimensions across items in a batch.

## PatchCB
- Converts batch ([bs x seq_len x n_vars]) to patched input ([bs x num_patch x n_vars x patch_len])
- Overwrites learners input attribute to use the patched input.

## TrackTimerCB
- Records epoch start and end time.

## TrackTrainingCB
- Tracks training/validation metrics.

## TrackerCB
- Checks if current epoch has improved on previous epoch results

## PrintResultsCB
- Prints results to standard output

## OneCycleLR
- Sets up a learning rate scheduler of type `torch.optim.lr_scheduler`, and advances through it after each batch.

