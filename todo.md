# To do in the future

---

## Regarding the code in general

- Shorter functions, shorter files.
- Magic values are a bit everywhere, we should refactor them into config variables or visible constants
- Better structure of the repo, with more modularity and clean modules.
- Build pydantic models to properly define the configs, input and output structures (e.g. dict, dataframes, ...).

---

## Regarding setup and coding experience

- Add installation scripts, running scripts.
- Add config handling (e.g. Hydra)
- Automation, orchestration and monitoring of different parts of the pipeline (e.g. data processing, model training, evaluation).
- Add a CI (precommit hooks, tests, etc.)

---

## Regarding scalability

The current implementation is not scalable. This should be handled at several levels

### Data

- **Data preprocessing:** we manipulate different directories or files manually, without real acceleration, streaming nor parallelism. The first basic manipulations (e.g. merging files, moving or splitting them, changing column names, etc.) should leverage parallelism and streaming. Then, standard preprocessing should leverage CPU/GPU parallelism as well as streaming computations based on optimized frameworks (e.g. polars, RAPIDS, etc.).
- **Data loading:** torch datasets and dataloaders should be optimized to properly fit the available memory based on streaming data loading.

### Model

- At a low level, the model is coded by hand, it does not benefit from FlashAttention or other optimized implementations. 
- At a higher level, we should implement proper data parallelism (e.g. DeepSpeed, DDP) to train the model on multiple GPUs. Given the relatively little size of TS models, model parallelism seems overkilled. 

### Tests

Implement tests with mock models and datasets.

### Experiments

- Experiments should be properly tracked and monitored using for instance Neptune ;) (save metrics, artifacts, checkpoints, handle failures, etc.)
- Large-scale parallelization of experiments should rely on Docker and higher level orchestration frameworks.

---

## Regarding the data

### Better data analysis

Data should be appropriately analyzed and dispatched in order to measure generalization and transferability of the model, depending on e.g:

- domain. Data in the same domain should share similar characteristics, so we could expect a similar behavior of the model in the same domain.
- frequency. 
- seasonality. 
- size imbalance, to measure the influence of large datasets compared to small ones.

### Preprocessing analysis

We should refine:

- NaN value handling (currently just discarding the series with NaN values)
- outliers handling

We should analyze the impact of:

- the different patching sizes
- the window size sampling (context and prediction length)
- adding covariates (e.g. past_feat_dynamic_real)

---

## Regarding the model

This is a simplified and minimal implementation of the MOIRAI model. It should be refined to meet the original one. Experiments and ablations could also be performed (e.g. new version with MoE architecture, normalization, output distributions, etc). 

We should also implement an hyperparameter search process.

---

## Regarding the evaluation

Current evaluation is naive, both in conception and implementation. With more time on the project, I would probably focus on this part. In particular, 
- I would like to build a proper evaluation dataset and pipeline to load in memory a single file of several datasets at evaluation time. 
- The datasets should be chosen based on their similarity to the training datasets, and we should compare the performance based on several characteristics (e.g. domain, frequency, seasonality, size imbalance, etc.).
- We should properly evaluate the performances w.r.t. context and prediction length.
- We should compare in depth the time-related features of the predictions and the targets.