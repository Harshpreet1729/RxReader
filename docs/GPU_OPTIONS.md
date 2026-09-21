# GPU options

The included workflow targets a CUDA GPU in Google Colab; a T4 is the initial small-batch target, not a tested universal memory guarantee. CPU is supported for inference but rejected by the training command.

- [Colab](https://research.google.com/colaboratory/faq.html): limited free GPU access; availability and session limits vary. Requires a Google account; the notebook uses Drive authorization for persistent checkpoints.
- [Kaggle](https://www.kaggle.com/docs/efficient-gpu-usage): free notebook GPU quotas subject to current account eligibility and availability.
- [Runpod](https://www.runpod.io/pricing) and [Vast.ai](https://vast.ai/pricing): paid GPU rental options. Review live GPU, storage, transfer, tax and currency-conversion charges before renting.

No cost or training-duration guarantee is made. Start with an available free runtime; estimate your own throughput before paying for a longer experiment. Save the model and processor together before ending a session. This repository does not rent resources or submit billing information.
