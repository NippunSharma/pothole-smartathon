Installation steps:


1. git clone https://github.com/NippunSharma/pothole-smartathon
2. conda env create --file environment.yaml python=3.8.8
3. conda activate smartathon
4. pip install -r sort/requirements.txt
5. pip install -r yolov5/requirements.txt

After installation, you can download the demo folder from [here]() and place in
the root directoty of the project.

You can also download the results folder from [here]() and place in the root directory
of the ptoject.

If there is some issue while installing certain pip packages, please make sure
that you have upgraded to a new version of pip. This can be done by:

```bash
pip install --upgrade pip
```


Please note that all pretrained models are present inside the  `pretrained_model` folder.
The models are exported in both `onnx` as well as `torchscript` format.

Please see `pothole_analysis.ipynb` for the detailed description of my approach.



In case of any queries, kindly contact me at: [inbox.nippun@gmail.com](mailto:inbox.nippun@gmail.com)