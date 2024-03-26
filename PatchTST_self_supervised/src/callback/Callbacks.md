# Callbacks


## SetupLearnerCB
- Moves model to device before fitting
- Moves data to device before each batch train/val/test step.
- Decide is GPU if available else CPU


## RevInCB
- Normalizes input before forward pass
- Denormalizes predictions after forward pass


## ObservationMaskCB
- Added by WFP APP-GRS
- Applies a mask over the `seq_len` x `n_vars` dimensions across items in a batch.