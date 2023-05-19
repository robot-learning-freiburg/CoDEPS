# CoDEPS
[**arXiv**](https://arxiv.org/abs/2303.10147) | [**Website**](http://codeps.cs.uni-freiburg.de/) | [**Video**](https://www.youtube.com/watch?v=4m4swaIkHyg)

This repository is the official implementation of the paper:

> **CoDEPS: Continual Learning for Depth Estimation and Panoptic Segmentation**
>
> [Niclas Vödisch](https://vniclas.github.io/), [Kürsat Petek](http://www2.informatik.uni-freiburg.de/~petek/), [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard/), and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada).
>
> *arXiv preprint arXiv:2303.10147*, 2023

<p align="center">
  <img src="codeps_overview.png" alt="Overview of CoDEPS approach" width="700" />
</p>

If you find our work useful, please consider citing our paper:
```
@article{voedisch23codeps,
  title={CoDEPS: Online Continual Learning for Depth Estimation and Panoptic Segmentation},
  author={Vödisch, Niclas and Petek, Kürsat and Burgard, Wolfram and Valada, Abhinav},
  journal={arXiv preprint arXiv:2303.10147},
  year={2023}
}
```


## 📔 Abstract

Operating a robot in the open world requires a high level of robustness with respect to previously unseen environments. Optimally, the robot is able to adapt by itself to new conditions without human supervision, e.g., automatically adjusting its perception system to changing lighting conditions. In this work, we address the task of continual learning for deep learning-based monocular depth estimation and panoptic segmentation in new environments in an online manner. We introduce CoDEPS to perform continual learning involving multiple real-world domains while mitigating catastrophic forgetting by leveraging experience replay. In particular, we propose a novel domain-mixing strategy to generate pseudo-labels to adapt panoptic segmentation. Furthermore, we explicitly address the limited storage capacity of robotic systems by proposing sampling strategies for constructing a fixed-size replay buffer based on rare semantic class sampling and image diversity. We perform extensive evaluations of CoDEPS on various real-world datasets demonstrating that it successfully adapts to unseen environments without sacrificing performance on previous domains while achieving state-of-the-art results.


## 👨‍💻 Code Release

We will make the code publicly accessible upon acceptance of our paper.


## 👩‍⚖️  License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
For any commercial purpose, please contact the authors.


## 🙏 Acknowledgment

This work was partly funded by the European Union’s Horizon 2020 research and innovation program under grant agreement No 871449-OpenDR and the Bundesministerium für Bildung und Forschung (BMBF) under grant agreement No FKZ 16ME0027.
<br><br>
<a href="https://opendr.eu/"><img src="./opendr_logo.png" alt="drawing" width="250"/></a>
