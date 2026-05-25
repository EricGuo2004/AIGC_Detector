# AIGC_Detector 10分钟课堂/答辩展示材料

> 使用说明：本文档用于直接制作 PPT 和练习口播，不生成 `.pptx` 文件。展示口径以 `README.md` 当前主结果为准；代码逻辑以 `train.py`、`test.py`、`src/data_utils.py`、`src/features.py`、`src/training.py`、`src/robustness.py` 的实际实现为准。  
> 资料口径提醒：`report/tables/final_result_selection.csv` 仍保留旧 baseline 选择表；当前展示建议引用 `README.md`、`report/tables/optimization_v2_summary.csv`、`report/submission_checklist.md` 中一致指向的 `outputs_v2_full_best`。

## 一、展示定位

### 中文标题建议

1. 频谱指纹能识别 AI 图片吗？
2. 从频域看 AI 图像的生成痕迹
3. 不看画面内容，能检测 AI 图片吗？

### 一句话项目介绍

本项目使用 FFT/DCT 等频域特征和 LightGBM，对 GenImage 中的 AI 图像与真实图像进行检测，并进一步判断 AI 图像来自 ADM、BigGAN、VQDM、GLIDE 等生成源。

### 10分钟展示核心主线

AI 图像可能在频域留下生成器指纹；本项目把图像变成可解释频谱特征，再用轻量模型完成检测和归因。

### 主要依据文件

- 当前主结果：`README.md`
- 数据与训练入口：`train.py`、`test.py`
- 数据发现：`src/data_utils.py`
- 频域特征：`src/features.py`
- 模型训练与指标：`src/training.py`
- 鲁棒性评估：`src/robustness.py`、`scripts/evaluate_best_robustness.py`
- 报告素材生成：`scripts/build_report_assets.py`
- 报告图表：`report/figures/`
- 报告表格：`report/tables/`

## 二、PPT逐页内容

### Slide 1：封面

**页面文字**

- AIGC 频谱指纹检测
- AI vs Nature
- 生成源归因
- FFT / DCT / LightGBM
- 作者：填入姓名

**推荐视觉元素**

- 背景可放一张 AI 图像与真实图像对比示意。
- 如果想更贴合项目，可放 `report/figures/spectrum_examples.png` 的局部截图。

**逐字讲稿（约50秒）**

大家好，我今天展示的项目是 AIGC 图像的频谱指纹检测。简单说，我们不是直接看图片画得像不像，而是把图片转换到频域，看它里面不同频率成分的分布。这个项目做了两个主要任务：第一，判断一张图是 AI 生成还是自然真实图像；第二，如果它是 AI 图像，进一步判断它来自 ADM、BigGAN、VQDM 还是 GLIDE。方法上，我们使用 FFT、DCT 等可解释频域特征，再用 LightGBM 这种轻量模型完成分类。

**预计用时**

50 秒

**转场句**

- 上一页到本页：无，开场直接进入。
- 本页到下一页：为什么要做这个检测？先从问题背景说起。

### Slide 2：问题背景

**页面文字**

- AI 图片越来越真
- 肉眼判断不稳定
- 误用风险增加
- 需要可解释检测
- 不能只看外观

**推荐视觉元素**

- 左右对比：真实图像 vs AI 图像。
- 可用简单问号图标表示“肉眼难判断”。

**逐字讲稿（约60秒）**

现在 AI 生成图片的质量越来越高，很多时候只看画面内容，已经很难稳定判断真假。这个问题很重要，因为 AI 图片可能出现在新闻、社交媒体、作业作品、商业素材审核等场景里。如果检测方法只是说“模型觉得像 AI”，但讲不清依据，答辩或者实际使用时都会比较弱。所以这个项目选择了一个更可解释的角度：不只看像素外观，而是看图像生成过程可能留下的统计痕迹。

**预计用时**

60 秒

**转场句**

- 上一页到本页：先说明检测为什么有意义。
- 本页到下一页：那这些统计痕迹在哪里？核心就在频域。

### Slide 3：核心想法

**页面文字**

- 图像也有“频谱”
- 低频是轮廓
- 高频是细节
- AI 可能留指纹
- 用统计特征捕捉

**推荐视觉元素**

- 首选：`report/figures/spectrum_examples.png`
- 可以加一句小字：图片像声音，频域像看高低音成分。

**逐字讲稿（约65秒）**

频域可以用声音来类比。一个声音可以分解成高音和低音，一张图像也可以分解成不同频率的成分。低频大致对应整体明暗和大轮廓，高频更像边缘、纹理、噪声和细节。真实相机图像和 AI 生成图像的形成过程不同，比如生成器上采样、解码、去噪和后处理方式不同，所以它们可能在高频、方向纹理、颜色通道频谱或者 DCT 块结构上留下差异。本项目就是把这些差异变成一组可解释特征。

**预计用时**

65 秒

**转场句**

- 上一页到本页：检测的依据不是语义，而是频谱结构。
- 本页到下一页：基于这个想法，项目具体定义了三个任务。

### Slide 4：任务定义

**页面文字**

- 任务1：AI / Nature
- 任务2：生成源归因
- 来源：四个生成器
- 诊断：鲁棒性评估
- 指标：Acc / F1 / AUC

**推荐视觉元素**

- 三块任务卡片：二分类、四分类归因、鲁棒性。
- 任务 2 标签写 ADM、BigGAN、VQDM、GLIDE。

**逐字讲稿（约60秒）**

项目里有两个正式分类任务和一个诊断任务。第一个是 AI vs Nature 二分类，标签就是 `ai` 和 `nature`，判断图片是不是 AI 生成。第二个是 AI 生成源归因，只在 AI 图片里判断它来自 ADM、BigGAN、VQDM 还是 GLIDE。第三个是鲁棒性评估，也就是图片经过 JPEG 压缩、resize 和 Gaussian noise 之后，模型还能不能稳定工作。指标上，二分类看 Accuracy、Macro-F1 和 AUC；多分类归因主要看 Accuracy 和 Macro-F1。这里 Macro-F1 很重要，因为它会平等考虑每个类别。

**预计用时**

60 秒

**转场句**

- 上一页到本页：先把项目目标讲清楚。
- 本页到下一页：接下来看看数据怎么组织，代码怎么支撑这些任务。

### Slide 5：数据与代码结构

**页面文字**

- GenImage 四个源
- train 用于训练
- val 用作测试
- 每源含 ai/nature
- 代码模块清晰

**推荐视觉元素**

- 数据目录树：
  - `ADM/train/ai`
  - `ADM/train/nature`
  - `ADM/val/ai`
  - `ADM/val/nature`
- 可放 `report/tables/dataset_counts.csv` 摘要表。

**逐字讲稿（约60秒）**

数据使用 GenImage 中四个生成源：ADM、BigGAN、VQDM 和 GLIDE。每个生成源下面都有 `train/ai`、`train/nature`、`val/ai`、`val/nature`。按照 README 和代码实现，`train` 用于训练，`val` 在项目里作为测试集使用。代码结构也比较直接：`src/data_utils.py` 负责扫描路径并生成样本标签；`src/features.py` 负责把图像变成频域特征；`src/training.py` 负责训练 LightGBM 等模型并保存指标；`src/robustness.py` 负责对 JPEG、resize 和 noise 做鲁棒性评估。

**预计用时**

60 秒

**转场句**

- 上一页到本页：任务确定以后，要看数据和代码入口。
- 本页到下一页：有了数据，下面就是最核心的方法流程。

### Slide 6：方法流程

**页面文字**

- 输入图像
- 读取样本标签
- 提取频域特征
- 训练 LightGBM
- 输出指标图表

**推荐视觉元素**

- 建议生成/建议展示项目流程图：  
  `图片输入 -> data_utils -> features -> LightGBM -> 指标/图表`
- 可直接复制下面 Mermaid 到制图工具：

```mermaid
flowchart LR
  A[图片输入] --> B[data_utils 读取]
  B --> C[FFT/DCT 特征]
  C --> D[LightGBM 训练]
  D --> E[Accuracy / F1 / AUC]
  E --> F[鲁棒性与报告图表]
```

**逐字讲稿（约75秒）**

整个流程可以拆成五步。第一步输入图片。第二步，`data_utils.py` 根据目录结构读取样本，二分类时标签是 `ai` 和 `nature`，归因时标签是 ADM、BigGAN、VQDM、glide。第三步，`features.py` 提取频域特征。基础特征包括 FFT 径向功率谱、角向能量、低中高频能量比例、高频统计、patch 高频变化和 DCT 径向谱。当前最佳路线使用 `fusion_freq`，进一步加入 RGB/YCbCr 颜色频谱、多尺度频谱、8x8 block DCT 和残差频谱。第四步，`training.py` 使用 LightGBM 训练并选择模型。第五步，输出模型、指标、混淆矩阵、feature importance 和预测明细。

**预计用时**

75 秒

**转场句**

- 上一页到本页：现在把模块串起来看整体流程。
- 本页到下一页：这条流程在 clean validation 上效果如何？看当前 README 结果。

### Slide 7：实验结果

**页面文字**

- 主结果：v2 full best
- 二分类 F1：0.9913
- 二分类 AUC：0.9996
- 归因 F1：0.9991
- clean 表现很强

**推荐视觉元素**

- 指标结果表，直接引用 `README.md` 当前结果：

| Task | Accuracy | Macro-F1 | AUC |
| --- | ---: | ---: | ---: |
| AI vs Nature | 0.9913 | 0.9913 | 0.9996 |
| Source attribution | 0.9991 | 0.9991 | - |

- 可选图：`report/figures/optimization_v2_macro_f1.png`
- 可选混淆矩阵：
  - `report/figures/confusion_binary_ai_vs_nature_lightgbm.png`
  - `report/figures/confusion_ai_subsource_attribution_lightgbm.png`

**逐字讲稿（约85秒）**

当前 README 中的主结果是 `outputs_v2_full_best`，配置是 `fusion_freq + flat LightGBM + wide profile`，训练增强为 none。clean validation 上，AI vs Nature 二分类的 Accuracy 和 Macro-F1 都是 0.9913，AUC 是 0.9996。Source attribution 的 Accuracy 和 Macro-F1 都是 0.9991。这里 Accuracy 表示整体判断对了多少；Macro-F1 更强调各个类别是否都被公平地分好；AUC 只用于二分类，表示模型区分 AI 和 nature 的排序能力。这个结果说明，在当前 GenImage 四个生成源的 train/val 协议下，频域融合特征非常有效。

**预计用时**

85 秒

**转场句**

- 上一页到本页：方法流程跑完以后，先看最核心的 clean 指标。
- 本页到下一页：但 clean 很强不等于真实场景完全可靠，下面看鲁棒性。

### Slide 8：鲁棒性与局限

**页面文字**

- 后处理会降分
- JPEG 破坏频谱
- resize 改变采样
- noise 污染高频
- clean 不等于可靠

**推荐视觉元素**

- 首选鲁棒性图：
  - `report/figures/robustness_binary_ai_vs_nature.png`
  - `report/figures/robustness_ai_subsource_attribution.png`
- 对比图：
  - `report/figures/robustness_comparison_binary_ai_vs_nature.png`
  - `report/figures/robustness_comparison_ai_subsource_attribution.png`
- 可放小表：

| 条件 | Binary F1 | Attribution F1 |
| --- | ---: | ---: |
| clean | 0.9908 | 0.9992 |
| JPEG 95 | 0.6607 | 0.2952 |
| resize 0.5 | 0.3671 | 0.1551 |
| noise 10 | 0.3377 | 0.1000 |

**逐字讲稿（约85秒）**

鲁棒性是这个项目最重要的限制。`outputs_v2_full_best_robust_20pct` 在 20% validation 子集上做了 JPEG、resize 和 Gaussian noise 评估。二分类在 clean 子集上的 Macro-F1 是 0.9908，但 JPEG 95 后降到 0.6607，resize 0.5 后降到 0.3671，noise 10 后降到 0.3377。归因任务更敏感，clean 是 0.9992，但 JPEG 95 后只有 0.2952，noise 10 时是 0.1000。原因是当前最强的 `fusion_freq` 很依赖颜色频谱、高频统计和块频率结构，而这些正好容易被压缩、重采样和噪声改变。所以展示时要强调：clean 结果很强，但真实场景还需要更系统的鲁棒训练和跨生成器验证。

**预计用时**

85 秒

**转场句**

- 上一页到本页：clean 分数高以后，要主动说明风险。
- 本页到下一页：最后总结这个项目做成了什么，以及后续怎么改进。

### Slide 9：总结与未来工作

**页面文字**

- 频域特征有效
- LightGBM 轻量可解释
- clean 指标很强
- 鲁棒性仍不足
- 未来增强泛化

**推荐视觉元素**

- 三栏总结：贡献、局限、未来工作。
- 可选图：`report/figures/feature_importance_binary_ai_vs_nature_top20.png`
- 答辩备用图：`report/figures/logo_generalization_macro_f1.png`

**逐字讲稿（约60秒）**

总结一下，这个项目证明了一个比较轻量但有效的思路：把 AIGC 检测转成可解释的频域统计问题。当前最佳路线 `fusion_freq + LightGBM` 在 clean validation 上二分类 Macro-F1 达到 0.9913，生成源归因 Macro-F1 达到 0.9991。它的优势是特征含义清楚、训练相对轻量、可以输出混淆矩阵和特征重要性。主要局限是对 JPEG、resize 和噪声比较敏感，跨生成器泛化也不均匀。后续可以把鲁棒增强、stable 频域特征、leave-one-generator-out 验证和更多真实数据去重检查加入模型选择流程。

**预计用时**

60 秒

**转场句**

- 上一页到本页：最后收束到贡献和限制。
- 本页到结束：我的展示到这里，谢谢大家，欢迎提问。

## 三、时间控制

总时长建议控制在 10 分钟左右，允许在 9 到 11 分钟之间浮动。

| Slide | 标题 | 预计秒数 |
| --- | --- | ---: |
| 1 | 封面 | 50 |
| 2 | 问题背景 | 60 |
| 3 | 核心想法 | 65 |
| 4 | 任务定义 | 60 |
| 5 | 数据与代码结构 | 60 |
| 6 | 方法流程 | 75 |
| 7 | 实验结果 | 85 |
| 8 | 鲁棒性与局限 | 85 |
| 9 | 总结与未来工作 | 60 |
| **总计** |  | **600 秒 / 10 分钟** |

压缩到 9 分钟的方法：

- Slide 3 少讲一个频域例子，省 15 秒。
- Slide 6 少讲 block DCT 和残差频谱细节，省 20 秒。
- Slide 8 只讲 3 个鲁棒性数字，省 25 秒。

扩展到 11 分钟的方法：

- Slide 7 加一张混淆矩阵解释错分。
- Slide 9 简单补充 leave-one-generator-out 的泛化风险。

## 四、PPT文案风格提醒

- 每页正文控制在 5 个要点以内。
- 每个要点尽量短，避免整句堆满页面。
- 对小白讲频域时，用“图片像声音，频域像高低音成分”类比。
- 不需要推公式，FFT/DCT 只讲直觉：把图像拆成不同频率。
- 结果页重点讲指标含义，不要只报数字。

## 五、图表与素材建议

### 已存在、推荐使用的图表

| 建议页码 | 素材路径 | 用途 |
| --- | --- | --- |
| Slide 3 | `report/figures/spectrum_examples.png` | 展示 AI / Nature 频谱示意 |
| Slide 5 | `report/tables/dataset_counts.csv` | 展示 GenImage 四个生成源数据规模 |
| Slide 7 | `report/figures/optimization_v2_macro_f1.png` | 展示 `fusion_freq` 路线提升 |
| Slide 7 | `report/figures/confusion_binary_ai_vs_nature_lightgbm.png` | 二分类混淆矩阵 |
| Slide 7 | `report/figures/confusion_ai_subsource_attribution_lightgbm.png` | 生成源归因混淆矩阵 |
| Slide 8 | `report/figures/robustness_binary_ai_vs_nature.png` | 二分类鲁棒性曲线 |
| Slide 8 | `report/figures/robustness_ai_subsource_attribution.png` | 归因鲁棒性曲线 |
| Slide 8 | `report/figures/robustness_comparison_binary_ai_vs_nature.png` | 多模型鲁棒性对比 |
| Slide 9 | `report/figures/feature_importance_binary_ai_vs_nature_top20.png` | 特征重要性解释 |
| Q&A备用 | `report/figures/logo_generalization_macro_f1.png` | 回答新生成器泛化问题 |

### 已存在、推荐引用的表格

| 表格路径 | 用途 |
| --- | --- |
| `report/tables/optimization_v2_summary.csv` | 当前 `outputs_v2_full_best` 主结果 |
| `report/tables/robustness_binary_ai_vs_nature.csv` | 二分类鲁棒性数值 |
| `report/tables/robustness_ai_subsource_attribution.csv` | 归因鲁棒性数值 |
| `report/tables/top_features_binary_ai_vs_nature.csv` | 二分类 Top 特征 |
| `report/tables/top_features_ai_subsource_attribution.csv` | 归因 Top 特征 |
| `report/tables/logo_generalization.csv` | 留一生成器泛化结果 |

### 建议生成/建议展示但当前没有独立文件的素材

- `project_pipeline_flow.png`：展示 `图片输入 -> data_utils -> features -> LightGBM -> 指标/鲁棒性`。
- `readme_result_table.png`：把 README 当前 clean validation 指标做成 PPT 表格。
- `robustness_key_numbers.png`：只展示 clean、JPEG 95、resize 0.5、noise 10 四个关键条件。

### PPT 中至少建议使用的四类素材

1. 项目流程图：建议自己生成，见 Slide 6 Mermaid。
2. 指标结果表：直接引用 `README.md` 当前 clean validation 结果。
3. 鲁棒性对比图或表：使用 `report/figures/robustness_binary_ai_vs_nature.png` 或 Slide 8 小表。
4. 特征重要性图或频谱示意图：使用 `report/figures/spectrum_examples.png` 或 `feature_importance_*_top20.png`。

## 六、完整连续口播讲稿

大家好，我今天展示的项目是 AIGC 图像的频谱指纹检测。这个项目想回答一个问题：如果 AI 图片越来越真实，我们还能不能不靠肉眼，而是靠图像内部的统计痕迹来判断它是不是 AI 生成？

现在 AI 生成图片的质量已经很高，很多时候只看画面内容，很难稳定判断真假。这个问题在新闻、社交媒体、课程作业、素材审核这些场景里都很重要。而且我希望这个检测方法不只是给出一个黑盒结果，还能说清楚它依据了什么。因此本项目没有直接训练 CNN 或 ViT，而是选择从频域入手。

频域可以用声音来类比。一个声音可以拆成高音和低音，一张图片也可以拆成不同频率的成分。低频通常对应整体明暗和大轮廓，高频对应边缘、纹理、细节和噪声。真实相机图像和 AI 生成图像的形成过程不同，比如生成器的上采样、解码、去噪和后处理方式都可能留下痕迹。这些痕迹肉眼不一定明显，但可能在频谱里表现为稳定的统计差异。

具体来说，本项目有两个分类任务和一个诊断任务。第一个任务是 AI vs Nature 二分类，也就是判断图片是 AI 生成还是自然真实图像。第二个任务是生成源归因，只看 AI 图像，进一步判断它来自 ADM、BigGAN、VQDM 还是 GLIDE。第三个任务是鲁棒性评估，检查模型在 JPEG 压缩、resize 和 Gaussian noise 后还能不能稳定工作。

数据来自 GenImage 的四个生成源。每个生成源下面都有 `train/ai`、`train/nature`、`val/ai` 和 `val/nature`。按照当前 README 和代码实现，`train` 用来训练，`val` 在项目里作为测试集使用。代码结构也比较清楚：`src/data_utils.py` 负责读取数据和生成标签，`src/features.py` 负责提取 FFT、DCT 等频域特征，`src/training.py` 负责训练 LightGBM 并输出指标，`src/robustness.py` 负责做后处理扰动评估。

方法流程可以概括为五步。首先输入图像；然后 `data_utils` 根据目录结构读取样本；接着 `features.py` 把图像转成特征。基础特征包括 FFT 径向功率谱、角向能量、低中高频能量比例、高频统计、patch 高频变化和 DCT 径向谱。当前最好的路线叫 `fusion_freq`，它还加入了 RGB 和 YCbCr 颜色频谱、多尺度频谱、8x8 block DCT，以及 median、smooth、laplace 残差频谱。然后用 LightGBM 训练模型，最后输出 Accuracy、Macro-F1、AUC、混淆矩阵和 feature importance。

从当前 README 的结果看，主结果是 `outputs_v2_full_best`，配置是 `fusion_freq + flat LightGBM + wide profile`。在 clean validation 上，AI vs Nature 二分类 Accuracy 和 Macro-F1 都是 0.9913，AUC 是 0.9996。生成源归因任务 Accuracy 和 Macro-F1 都是 0.9991。这说明在当前 GenImage 四个生成源的 train/val 协议下，频域融合特征已经非常强。

不过，这里不能只讲 clean 分数。鲁棒性评估显示，模型对后处理比较敏感。在 20% validation 子集上，二分类 clean Macro-F1 是 0.9908，但 JPEG 95 后降到 0.6607，resize 0.5 后降到 0.3671，noise 10 后降到 0.3377。归因任务下降更明显，noise 10 时 Macro-F1 只有 0.1000。原因是当前最有效的特征正好依赖颜色频谱、高频统计和块频率结构，而这些会被压缩、重采样和噪声改变。

所以这个项目的结论是：频域指纹确实能有效检测当前数据协议下的 AIGC 图像，并且比黑盒视觉模型更容易解释；但它还不能直接等同于真实场景完全可靠。未来可以做三方面改进：第一，把 JPEG、resize、noise、blur、crop 等扰动系统加入训练；第二，把 `stable_freq` 这类更稳定的特征和 `fusion_freq` 融合；第三，把 leave-one-generator-out 这类跨生成器泛化测试纳入模型选择。我的展示到这里，谢谢大家。

## 七、答辩 Q&A

### Q1：为什么不用 CNN 或 ViT？

答：这个项目的重点不是追求黑盒 SOTA，而是验证“频域指纹是否可检测、可解释”。CNN 或 ViT 可能效果很好，但更难说明它到底看了什么。这里的 FFT、DCT、颜色频谱、block DCT、高频统计都能对应具体信号含义，还能用 `feature_importance.csv` 解释模型关注的特征。

### Q2：为什么频域能检测 AI 图片？

答：AI 图像和真实相机图像的形成过程不同。生成器的上采样、解码、去噪、压缩和纹理合成方式，可能改变高频能量、方向纹理、颜色通道频谱和 DCT 块结构。频域正好适合观察这些肉眼不明显但统计上稳定的差异。

### Q3：clean accuracy 很高是不是过拟合？

答：有这个风险，所以不能只看 clean 指标。项目额外做了鲁棒性评估和 leave-one-generator-out 泛化实验。结果显示 clean 很强，但后处理和未见生成器会造成明显下降。因此展示里应把 clean 结果说成“当前 GenImage train/val 协议下很强”，而不是说真实场景已经完全可靠。

### Q4：怎么避免数据泄漏？

答：当前代码层面，`src/data_utils.py` 按 `train` 和 `val` 分开读取，`train.py` 和 `test.py` 都把 `train` 用于训练、`val` 用于测试。鲁棒性评估也加载保存好的模型，在 validation 子集上单独做扰动，不参与训练。更严格的后续改进是加入文件哈希去重、近重复图像检测，以及跨生成器留一验证。

### Q5：为什么鲁棒性会下降？

答：因为频域特征对图像后处理敏感。JPEG 会改变 DCT 和高频分布，resize 会重采样并改变频谱结构，Gaussian noise 会直接污染高频统计。当前最佳 `fusion_freq` 很依赖这些频域痕迹，所以 clean 很强，但遇到后处理会明显偏移。

### Q6：LightGBM 的优势是什么？

答：LightGBM 对表格特征很适合，训练快，能处理非线性特征组合，也能输出 feature importance。这个项目的输入不是原始像素，而是一组结构化频域特征，所以 LightGBM 比直接上深度视觉模型更轻量，也更容易解释。

### Q7：Source attribution 和二分类有什么区别？

答：二分类只判断图片是 AI 还是 nature，标签是 `ai` 和 `nature`。Source attribution 只看 AI 图像，进一步判断它来自 ADM、BigGAN、VQDM 还是 GLIDE，是四分类任务。归因更像是在识别不同生成器的“频谱签名”。

### Q8：这个方法面对新生成器还能不能用？

答：不能保证直接稳定。`report/tables/logo_generalization.csv` 的留一生成器实验显示，held-out GLIDE 的 Macro-F1 约 0.8877，但 held-out VQDM 只有约 0.4386，说明模型既学到了一些通用 AIGC 痕迹，也依赖已见生成器的特定频谱。面对新生成器，应该重新做跨域验证或加入更多生成器训练。

### Q9：后续如何改进？

答：可以从四个方向改进：第一，训练阶段加入更系统的 JPEG、resize、noise、blur、crop 扰动；第二，融合 `stable_freq` 和 `fusion_freq`，平衡 clean 性能和 degraded 稳定性；第三，把 leave-one-generator-out 作为模型选择标准；第四，引入更多真实数据、更多生成器和文件去重检查。

### Q10：为什么归因任务没有 AUC？

答：当前 `src/training.py` 只在二分类且有概率输出时计算 AUC。归因是多分类任务，代码没有计算 multiclass AUC，所以 README 中 Source attribution 的 AUC 标为 `-`。展示时讲 Accuracy 和 Macro-F1 即可。

## 八、最后输出检查

- [x] 是否包含 9 页 PPT 内容
- [x] 是否包含逐页讲稿
- [x] 是否包含连续口播稿
- [x] 是否包含时间分配
- [x] 是否引用 README 当前指标
- [x] 是否包含鲁棒性和局限
- [x] 是否包含答辩 Q&A
- [x] 是否没有虚构不存在的文件或结果

