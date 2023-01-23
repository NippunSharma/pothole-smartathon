Explanation Video: [Click Here](https://vimeo.com/791501256)

Installation steps:


1. git clone https://github.com/NippunSharma/pothole-smartathon
2. conda env create --file environment.yaml python=3.8.8
3. conda activate smartathon
4. Now, install pytorch based upon your operating system and GPU. (refer from https://pytorch.org/get-started/locally/)
5. pip install -r sort/requirements.txt
6. pip install -r yolov5/requirements.txt

After installation, you can download the demo folder from [here](https://drive.google.com/drive/folders/1kbdq5wX7ZpVQtV8GDoySdapZK3cFw8du?usp=sharing) and place in
the root directoty of the project.

You can also download the results folder from [here](https://drive.google.com/drive/folders/1JiUqNV_Vhf_wKtu_thuAG0zm55kSNULs?usp=sharing) and place in the root directory
of the ptoject.

There is also a pdf version of the notebook in case latex does not render properly on your system.

If there is some issue while installing certain pip packages, please make sure
that you have upgraded to a new version of pip. This can be done by:

```bash
pip install --upgrade pip
```


Please note that all pretrained models are present inside the  `pretrained_model` folder.
The models are exported in both `onnx` as well as `torchscript` format.

Please see `pothole_analysis.ipynb` for the detailed description of my approach.

A screenshot of a pothole after reconstruction is present below:
![3d](https://github.com/NippunSharma/pothole-smartathon/blob/main/image_display.png)

As you can see, the distances measured are in a relative scale. If we multiplt these values
with an absolute scale (which was found to be approx. 0.27 in the  `pothole_analysis.ipynb`)
file, then the dimensions of the pothole would become: 0.27 * 1.33 meter (=36 cm) and 0.27 * 0.41 meter (=11cm)

In case of any queries, kindly contact me at: [inbox.nippun@gmail.com](mailto:inbox.nippun@gmail.com)
