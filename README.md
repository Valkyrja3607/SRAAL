# SRAAL in PyTorch
 Reproduction implementation of SRAAL([State-Relabeling Adversarial Active Learning](https://github.com/Valkyrja3607/survey/issues/4))

### Prerequisites:
- Linux or macOS
- Python 3.8
- CPU compatible but NVIDIA GPU + CUDA CuDNN is highly recommended.

### Experiments and Visualization
The code can simply be run using 
```
python3 main.py
```
If you want to use GPU
```
python3 main.py --cuda
```
When using the model with different datasets or different variants, the main hyperparameters to tune are
```
--adversary_param --beta --num_vae_steps and --num_adv_steps
```

The results will be saved in `results/accuracies.log`. 
