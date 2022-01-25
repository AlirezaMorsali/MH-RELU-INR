# MH-RELU-INR
This is the Tensorflow 2.x implementation of our paper ["Multi-Head ReLU Implicit Neural Representation Networks"](https://arxiv.org/abs/2110.03448), accepted in ICASSP 2022. 

<div align=center>
<img width=95% src="https://github.com/AlirezaMorsali/MH-RELU-INR/blob/main/pics/Architecture.png"/>
</div>
In this paper, a novel multi-head multi-layer perceptron (MLP) structure is presented for implicit neural representation (INR). Since conventional rectified linear unit (ReLU) networks are shown to exhibit spectral bias towards learning low-frequency features of the signal, we aim at mitigating this defect by taking advantage of local structure of the signals. To be more specific, an MLP is used to capture the global features of the underlying generator function of the desired signal. Then, several heads are utilized to reconstruct disjoint local features of the signal, and to reduce the computational complexity, sparse layers are deployed for attaching heads to the body. Through various experiments, we show that the proposed model does not suffer from the special bias of conventional ReLU networks and has superior generalization capabilities. Finally, simulation results confirm that the proposed multi-head structure outperforms existing INR methods with considerably less computational cost.

## Convergence 
https://user-images.githubusercontent.com/30603302/148702625-6e69f8e0-e631-4b63-86d4-a379e2b27eb0.mp4

# Run
## Demo
Run experiments on Google Colab:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlirezaMorsali/MH-RELU-INR/blob/master/Demo_MH_RELU_INR.ipynb)

## 1. Clone Repository
```bash
$ git clone https://github.com/AlirezaMorsali/MH-RELU-INR.git
$ cd MH-RELU-INR/
```
## 2. Requirements
- Tensorflow >= 2.3.0
- Numpy >= 1.19.2
- Scikit-image >= 4.50.2
- Matplotlib> = 3.3.1
- Opencv-python >= 4.5.1

```bash
$ pip install -r requirements.txt
```

## 3. Set hyperparameters and training config :
You only need to change the constants in the [hyperparameters.py](https://github.com/AlirezaMorsali/MH-RELU-INR/blob/main/hyperparameters.py) to set the hyperparameters and the training config.

## 4. Run experiments:
Use the following codes to run the experiments.
- Note : The results for the experiments are saved in the [result](https://github.com/AlirezaMorsali/MH-RELU-INR/tree/main/results) folder.

### Comparison experiments:
```bash
python run_comparison_experiments.py -i [path of input image] \
                                     -nh [number of heads for multi-head network] \
                                     -bh [root number of heads for base multi-head network(for fair comparison)] \
                                     -ba [alpha parameter for base multi-head network(for fair comparison)] \
                                     -ub [use bias for the head part of the multi-head network]
```
**Example:** 
```bash
python run_comparison_experiments.py -i pics/sample1.jpg \
                                     -nh 64 \
                                     -bh 64 \
                                     -ba 256 \
                                     -ub true
```
### Generalization experiments:
```bash
python run_generalization_experiments.py -i [path of input image] \
                                         -bh [root number of heads for base multi-head network(for fair comparison)] \
                                         -ba [alpha parameter for base multi-head network(for fair comparison)] \
                                         -ub [use bias for the head part of the multi-head network]
```
**Example:** 
```bash
python run_generalization_experiments.py -i pics/sample1.jpg \
                                         -bh 256 \
                                         -ba 32 \
                                         -ub false
```
### Spectral bias experiments:
```bash
python run_spectral_bias_experiments.py
```

## Citation

If you find our code useful for your research, please consider citing:
```bibtex
@article{aftab2021multi,
  title={Multi-Head ReLU Implicit Neural Representation Networks},
  author={Aftab, Arya and Morsali, Alireza},
  journal={arXiv preprint arXiv:2110.03448},
  year={2021}
}
```
