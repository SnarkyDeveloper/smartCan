# Journal

## Day one (00:30 - 01:10 GMT-7)
> [!TLDR] Initialized project structure without writing any training data
Created and started work on training code and compression into TensorLite to run on the Esp32-S3. None of this code is ready to train, dataset is still needed and so is configuration of more details. Code must also be reviewed again to ensure code works well.

## Day 2 (18:00-19:00 GMT-7)
> [!TLDR] Wrote actual project fitting logic and tests
Created fitting logic, saved the model as a .keras file for testing the model as well. Loaded in the actual dataset. Wrote proper testing with the `pytest` library. Training is done properly, although should be pushed a bit further as aiming for 93%+ accuracy. Currently at ~90% (Rounded by 100ths place). 

## Day 3 (22:00 - 23:00 GMT-7)
> [!TLDR] Updated accuracy to 91%
Switched to int categories, added validation sets and made model more trainable

> [!NOTE] Snarky out!
