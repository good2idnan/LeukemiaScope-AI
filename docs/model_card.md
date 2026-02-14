# Model Card: MedGemma Leukemia Detection LoRA

## Model Details

| Attribute      | Value                            |
| -------------- | -------------------------------- |
| **Model Name** | medgemma-1.5-4b-it-leukemia-lora |
| **Base Model** | google/medgemma-1.5-4b-it        |
| **Model Type** | LoRA Adapter                     |
| **Task**       | Binary Image Classification      |
| **Classes**    | Normal, Leukemia                 |

## Training Data

- **Dataset:** C-NMC Leukemia + Blood Cell Cancer (ALL)
- **Training Samples:** 9,701 images
- **Test Samples:** 1,867 images
- **Source:** [Kaggle](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)

## Performance Metrics

| Metric              | Value  |
| ------------------- | ------ |
| Accuracy            | 78.15% |
| Leukemia Precision  | 83%    |
| Leukemia Recall     | 83.10% |
| Normal Precision    | 68%    |
| Normal Recall       | 69%    |
| F1-Score (Leukemia) | 83.24% |

## Intended Use

This model is intended for:

- Educational purposes
- Research on medical AI
- Preliminary screening assistance

## Limitations

- NOT a diagnostic tool
- Trained on specific dataset style (cropped single cells)
- May not generalize to other imaging equipment
- Requires specialist confirmation for all results

## Ethical Considerations

- Model should augment, not replace, clinical judgment
- False negatives could delay treatment
- False positives could cause unnecessary anxiety

## Citation

```
@misc{leukemiascope2026,
  title={LeukemiaScope: AI Blood Cell Analysis},
  author={Team Name},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/good2idnan/medgemma-1.5-4b-it-leukemia-lora}
}
```
