# PointNet Segmentation on S3DIS Dataset
[Dataset](http://buildingparser.stanford.edu/dataset.html)
Download the dataset 'Stanford3dDataset_v1.2_Aligned_Version' in 'point_net' folder.
Also for testing on new scan (much different than dataset), I suggest to download scans from [here](https://www.ifi.uzh.ch/en/vmml/research/datasets.html), particularly scans in [here](https://files.ifi.uzh.ch/vmml/RoomsReconstructionDatasets/office1.zip), and place them in 'point_net' folder

## Requirements
* Pytorch with your favourite cuda
* Open3D




## Instructions

```bash
python s3dis_reducer.py
cd ..
cd point_net
python main_segment.py
```

## Ground Truth vs Predicted Point Cloud

<div style="display: flex; justify-content: space-around; align-items: center;">
    <figure>
        <img src="groundtruth.png" alt="Ground Truth Point Cloud" style="width: 90%;">
        <figcaption>Ground Truth</figcaption>
    </figure>
    <br>
    <br>
    <br>
    <figure>
        <img src="predicted.png" alt="Predicted Point Cloud" style="width: 90%;">
        <figcaption>Predicted</figcaption>
    </figure>
</div>

### Metrics

- **Accuracy:** 0.8587
- **MCC:** 0.8351
- **IOU:** 0.7523


## Testing on Unseen Point Cloud
<figure>
    <img src="ood_pred.png" alt="Segmentation of OOD Point Cloud" style="width: 100%;">
    <!-- <figcaption>Segmentation of OOD Point Cloud</figcaption> -->
</figure>
