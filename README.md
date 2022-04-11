# CUFD
Code of CUFD: An encoder-decoder network for visible and infrared image fusion based on common and unique feature decomposition

Tips
---------
#### To train:<br>
* Step1: download [Training dataset] or create your own training dataset [*your own training dataset*](https://github.com/hanna-xu/utils).
* Step2: In main.py, keep `IS_TRAINING==True` and choose the function train_part1.py (the 29th line in main.py), and then run main.py.
* Step3: In main.py, keep `IS_TRAINING==True` and choose the function generate_part1.py (the 30th line in main.py), and then run main.py.
* Step4: In main.py, keep `IS_TRAINING==True` and choose the function train_part2.py (the 31th line in main.py), and then run main.py.

#### To test with the pre-trained model:<br>
* Step1: download [feature dataset] which includes feature maps from I_e.
* Step2: In main.py, keep `IS_TRAINING==False`, and run main.py.

If this work is helpful to you, please cite it as:
```
@article{xu2022cufd,
  title={CUFD: An encoder--decoder network for visible and infrared image fusion based on common and unique feature decomposition},
  author={Xu, Han and Gong, Meiqi and Tian, Xin and Huang, Jun and Ma, Jiayi},
  journal={Computer Vision and Image Understanding},
  pages={103407},
  year={2022},
  publisher={Elsevier}
}
```
If you have any question, please email to me (meiqigong@whu.edu.cn).
