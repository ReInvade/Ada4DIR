
# Ada4DIR (IF 2025)
### 📖[**Paper**](https://www.sciencedirect.com/science/article/pii/S156625352500003X) | 🖼️[**PDF**](/figs/mains_1.png)

PyTorch codes for "[Ada4DIR: An adaptive model-driven all-in-one image restoration network for remote sensing images](https://www.sciencedirect.com/science/article/abs/pii/S1566253524000551)", **Information Fusion (IF)**, 2025.

- Authors:  [Ziyang LiHe](Ziyang_Lihe@whu.edu.cn), [Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/), [Jiang He*](https://jianghe96.github.io/), [Xianyu Jin](jin_xy@whu.edu.cn), [Yi Xiao](https://xy-boy.github.io/), [Yuzeng Chen](https://github.com/YZCU), [Huanfeng Shen](shenhf@whu.edu.cn), and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)<br>
- Wuhan University,


## Abstract
> Remote sensing images offer the opportunity to observe the Earth’s surface at multiple scales and from various angles. However, during acquisition, factors like blur, noise, haze, and low light can degrade the quality of optical remote sensing images. Deep learning-based image restoration methods are currently the most advanced approach for enhancing the usability of degraded remote sensing data. However, these methods are usually tailored to specific degradation types, which limits their effectiveness when faced with real-world degraded remote sensing data that may involve multiple, type-unknown degradation factors. In this paper, we model and derive solutions for four different types of remote sensing image degradation. With the integration of the novel prompt-injection-fusion block, the multi-level degradation information extraction capability of the multi-degradation-transformer block is further enhanced. Moreover, the model-driven prompt block in the 4D-fusion-transformer block enables adaptive recognition of different degradation types in remote sensing images and improves the physical interpretability of the restoration process. Finally, experimental results on three remote sensing image datasets and real-world multi-degradation image datasets demonstrate the advantages of the proposed network. 
## Network  
 ![image](/figs/mains_1.png)
 ![image](/figs/MDTBn4DTB.png)
 ![image](/figs/4D.png)


## 🎁 Dataset
Please download the following remote sensing benchmarks:
[MDRS]() 
### Train
```
python train.py
```

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: Ziyang_Lihe@whu.edu.cn

## Citation
If you find our work helpful in your research, please consider citing it. We appreciate your support！😊

```
@article{LIHE2025102930,
title = {Ada4DIR: An adaptive model-driven all-in-one image restoration network for remote sensing images},
journal = {Information Fusion},
volume = {118},
pages = {102930},
year = {2025},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2025.102930},
url = {https://www.sciencedirect.com/science/article/pii/S156625352500003X},
author = {Ziyang LiHe and Qiangqiang Yuan and Jiang He and Xianyu Jin and Yi Xiao and Yuzeng Chen and Huanfeng Shen and Liangpei Zhang},
keywords = {All-in-one, Model-driven network, Deep learning, Prompt learning, Remote sensing imaging},
abstract = {Remote sensing images offer the opportunity to observe the Earth’s surface at multiple scales and from various angles. However, during acquisition, factors like blur, noise, haze, and low light can degrade the quality of optical remote sensing images. Deep learning-based image restoration methods are currently the most advanced approach for enhancing the usability of degraded remote sensing data. However, these methods are usually tailored to specific degradation types, which limits their effectiveness when faced with real-world degraded remote sensing data that may involve multiple, type-unknown degradation factors. In this paper, we model and derive solutions for four different types of remote sensing image degradation. With the integration of the novel prompt-injection-fusion block, the multi-level degradation information extraction capability of the multi-degradation-transformer block is further enhanced. Moreover, the model-driven prompt block in the 4D-fusion-transformer block enables adaptive recognition of different degradation types in remote sensing images and improves the physical interpretability of the restoration process. Finally, experimental results on three remote sensing image datasets and real-world multi-degradation image datasets demonstrate the advantages of the proposed network. The source code and pre-trained models are available at https://github.com/colacomo/Ada4DIR.}
}
```
