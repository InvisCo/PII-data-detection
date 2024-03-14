# Results from Training Sessions

## Manual

| Model (Number of Documents) (Train-Valid-Test Split)               | Training Time * | F1-Score | Precision | Recall  | Notes                                                                        |
| ------------------------------------------------------------------ | --------------- | -------- | --------- | ------- | ---------------------------------------------------------------------------- |
| [6807 60-20-20](./trained_models/model-6807-60_20_20/)             | 70 min          | 0.58843  | 0.62021   | 0.55975 | Split train/test data differently                                            |
| [6807 60-20-20](./trained_models/model-6807-60_20_20/)             | 70 min          | 0.71074  | 0.77246   | 0.65816 | This model and below, split train test data with sklearn (matching baseline) |
| [1000 80(70-30)-20 A](./trained_models/model-1000-80(70_30)_20-A/) | 14 min          | 0.75482  | 0.82036   | 0.69898 | Learn rate = 0.001                                                           |
| [1000 80(70-30)-20 B](./trained_models/model-1000-80(70_30)_20-B/) | 11 min          | 0.61875  | 0.79839   | 0.5051  | Learn rate = 0.01                                                            |
| [1000 80(70-30)-20 C](./trained_models/model-1000-80(70_30)_20-C/) | 32 min          | 0.73889  | 0.81098   | 0.67857 | Learn rate = 0.0001                                                          |
| [1000 80(70-30)-20 D](./trained_models/model-1000-80(70_30)_20-D/) | 13 min          | 0.75     | 0.82317   | 0.68878 | Learn rate = 0.0005                                                          |

_\* Approximate time taken to train on a mobile RTX 3070Ti_

## Automated Results (Session A)
| Model                              | Training Time | F1-Score | Precision | Recall  | optimizer_learn_rate | training_dropout | batch_size_start | batch_size_stop | training_accumulate_gradient | training_patience | training_max_epochs | training_max_steps | training_eval_frequency | batch_size_compound | batcher_tolerance |
| ---------------------------------- | ------------- | -------- | --------- | ------- | -------------------- | ---------------- | ---------------- | --------------- | ---------------------------- | ----------------- | ------------------- | ------------------ | ----------------------- | ------------------- | ----------------- |
| [00](./trained_models/model-A-00/) | 08:05         | 0.69231  | 0.75      | 0.64286 | 0.001                | 0.1              | 100              | 1000            | 3                            | 1600              | 30                  | 20000              | 200                     | 1.001               | 0.2               |
| [01](./trained_models/model-A-01/) | 08:05         | 0.69638  | 0.76687   | 0.63776 | 0.001                | 0.18             | 100              | 1000            | 3                            | 1600              | 30                  | 20000              | 200                     | 1.001               | 0.2               |
| [02](./trained_models/model-A-02/) | 15:53         | 0.71257  | 0.86232   | 0.60714 | 0.001                | 0.26             | 100              | 1000            | 3                            | 1600              | 30                  | 20000              | 200                     | 1.001               | 0.2               |
| [03](./trained_models/model-A-03/) | 14:42         | 0.71341  | 0.88636   | 0.59694 | 0.001                | 0.34             | 100              | 1000            | 3                            | 1600              | 30                  | 20000              | 200                     | 1.001               | 0.2               |
| [04](./trained_models/model-A-04/) | 16:39         | 0.71687  | 0.875     | 0.60714 | 0.001                | 0.26             | 100              | 1000            | 3                            | 1600              | 30                  | 20000              | 200                     | 1.001               | 0.2               |

