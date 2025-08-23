

</div>
<div align="center">
<p align="center">
  <img src="assets/logo.png" width="10%" height="10%" />
</p>
</div>

<div align="center">
<h1>OmniThink</h1>
</div>
<div align="center">
<h3>Expanding Knowledge Boundaries in Machine Writing
through Thinking</h3>
</div>

<div align="center">


<!-- **Affiliations:** -->

👏 Welcome to try OmniThink in our **[<img src="./assets/tongyi.png" width="14px" style="display:inline;"> Modelscope online demo](https://www.modelscope.cn/studios/iic/OmniThink) and [🤗HuggingFace online demo]( https://huggingface.co/spaces/zjunlp/OmniThink)**!
<p align="center">
<a href="https://zjunlp.github.io/project/OmniThink">[🤖Project]</a>
<a href="https://arxiv.org/abs/2501.09751">[📄Paper]</a>
<a href="https://www.youtube.com/watch?v=5qQSJsiE0Sw&t=152s">[📺Youtube]</a> 

<!-- <a href="## 🚩Citation">[🚩Citation]</a> -->

</div>
<div align="center">
<p align="center">
  <img src="assets/overview.jpg" width="50%" height="50%" />
</p>
</div>

## Table of Contents
- 🚩[Acknowledgement](#Acknowledgement)
- 🌻[Quick Start](#quick-start)
- 🌟[Introduction](#Introduction)
- 🔧[Dependencies](#Dependencies)
- 📉[Results](#Results)
- 🧐[Evaluation](#evaluation)


# 🔔News
- `2025-08-24`, We have added local retrieval.
- `2025-03-12`, We have optimized the Docker usage for OmniThink.
- `2025-02-20`, We have added the evaluation methods from the paper to OmniThink, and in the future, we will integrate more evaluation methods.
- `2025-01-28`, We have provided support for the deepseek-reasoner model. You can try running ./examples/deepseekr1.py to test OmniThink's performance within deepseek-reasoner.
<details>
<summary><b>Previous News</b></summary>
  
- `2025-01-18`, we open-sourced OmniThink, a machine writing framework.

</details>


# 🌻Acknowledgement

- This work is implemented by [DsPY](https://github.com/stanfordnlp/dspy), [STORM](https://github.com/stanford-oval/storm) Sincere thanks for their efforts.
- We are also very grateful to [Zhangjiabao-nudt](https://github.com/Zhangjiabao-nudt) and [techshoww](https://github.com/techshoww) for their contributions to this repository.
- if you have any questions, please feel free to contact via xizekun.xzk@alibaba-inc.com, 1786594371@qq.com or xizekun2023@zju.edu.cn or create an issue.


## 📖 Quick Start

- 🌏 The **Online Demo** is avaiable at [ModelScope](https://www.modelscope.cn/studios/iic/OmniThink) now！


<img src="assets/demo.gif">

# 📌 Introduction

Welcome to **OmniThink**, an innovative machine writing framework designed to replicate the human cognitive process of iterative expansion and reflection in generating insightful long-form articles. 

- **Iterative Expansion and Reflection**: OmniThink uses a unique mechanism that simulates human cognitive behaviors to deepen the understanding of complex topics.
- **Enhanced Knowledge Density**: OmniThink focuses on expanding knowledge boundaries, resulting in articles that are rich in information and insights.
- **Comprehensive Article Generation**: OmniThink constructs outlines and generates articles, delivering high-quality content that is both coherent and contextually robust.
<div align="center">
    <img src="assets/main.jpg" width="80%" height="auto" />
</div>



# 🛠 Dependencies


## 📦 Conda

```bash
conda create -n OmniThink python=3.11
git clone https://github.com/zjunlp/OmniThink.git
cd OmniThink
# Install requirements
pip install -r requirements.txt
```

## 🐳 Docker
```
git clone https://github.com/zjunlp/OmniThink.git
docker pull zjunlp/omnithink:latest
docker run -it zjunlp/omnithink:latest
```

🔑 Before running, please export the LM API key and SEARCH key as an environment variable:


```bash
export LM_KEY=YOUR_API_KEY
export SEARCHKEY=YOUR_SEARCHKEY
```
> You can define your own [LM API](https://github.com/zjunlp/OmniThink/blob/main/src/tools/lm.py) and [SEARCH API](https://github.com/zjunlp/OmniThink/blob/main/src/tools/rm.py)

> Note that the output of the LM should be a LIST.

# Results in OmniThink
The preformance of OmniThink is shown below:
<div align="center">
    <img src="assets/table.jpg" width="95%" height="auto" />
</div>

# Generate Article in OmniThink
Just one command required
```bash
sh run.sh
```
You can find your Article, Outline and mindmap in ./results/

# 🔍 Evaluation

We provide convenient scripts for evaluating your method. The evaluation is divided into three categories: **Rubric_Grading**, **Knowledge_Density**, and **Information_Diversity**. 

We use the `factscore` library. Please run the following code before starting the evaluation.
```
cd eval
git clone https://github.com/shmsw25/FActScore.git
```

For Rubric Grading
 ```
 python Rubric_Grading.py \
  --articlepath articlepath \
  --modelpath modelpath
 ```

For Information Diversity
 ```
 python Information_Diversity.py \
  --mappath mappath \
  --model_path model_path
 ```

 For Knowledge_Density
 ```
 python Knowledge_Density.py \
  --articlepath articlepath \
  --api_path api_path \
  --threads threads
 ```


## Citation
If you find our repo useful in your research, please kindly consider cite:
```angular2
@misc{xi2025omnithinkexpandingknowledgeboundaries,
      title={OmniThink: Expanding Knowledge Boundaries in Machine Writing through Thinking}, 
      author={Zekun Xi and Wenbiao Yin and Jizhan Fang and Jialong Wu and Runnan Fang and Ningyu Zhang and Jiang Yong and Pengjun Xie and Fei Huang and Huajun Chen},
      year={2025},
      eprint={2501.09751},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.09751}, 
}
```

