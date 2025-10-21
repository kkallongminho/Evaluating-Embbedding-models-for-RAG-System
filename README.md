# Evaluating Domain-Specific Zero-Shot Retrieval without LLMs 🚀

This repository encapsulates the research from the paper "Evaluating Domain-Specific Zero-Shot Retrieval without LLMs: A Comparative Study of CSV-Derived and Benchmark Question Generation." Here you'll find the full LaTeX source, the compiled PDF, all experimental datasets, and the essential Python scripts used for model evaluation.

## Table of Contents 📖

- [Evaluating Domain-Specific Zero-Shot Retrieval without LLMs 🚀](#evaluating-domain-specific-zero-shot-retrieval-without-llms-%f0%9f%9a%80)
  - [Project Overview ✨](#project-overview-%e2%9c%a8)
  - [Research Goal 🎯](#research-goal-%f0%9f%8e%af)
  - [Methodology 🧪](#methodology-%f0%9f%a7%ac)
    - [Query Dataset Construction 📚](#query-dataset-construction-%f0%9f%93%93)
    - [Embedding Models and Metrics 📊](#embedding-models-and-metrics-%f0%9f%93%8a)
  - [Experimental Results 📈](#experimental-results-%f0%9f%93%88)
    - [HotpotQA Benchmark Performance 🏆](#hotpotqa-benchmark-performance-%f0%9f%8f%86)
    - [Domain-Specific Performance 🏢](#domain-specific-performance-%f0%9f%8f%a2)
  - [Discussion and Conclusion 🤔💡](#discussion-and-conclusion-%f0%9f%a4%af%f0%9f%92%a1)
  - [Project Structure 📁](#project-structure-%f0%9f%93%81)
  - [How to Reproduce 🛠️](#how-to-reproduce-%f0%9f%9b%a0%ef%b8%8f)
  - [Requirements ✅](#requirements-%e2%9c%85)
  - [License 📄](#license-%f0%9f%93%84)
  - [Contact 📧](#contact-%f0%9f%93%a7)

---

## Project Overview ✨

The rapid advancement of Large Language Models (LLMs) has redefined information retrieval, yet their computational demands remain a significant challenge. This research dives into **lightweight, embedding-based retrieval systems operating in a zero-shot setting**, an approach critical for developing efficient and grounded AI agents. Our central aim is to elucidate how diverse types of evaluation data—ranging from established benchmarks to pragmatic, domain-specific information—impact the perceived performance of these cutting-edge embedding models.

## Research Goal 🎯

Our primary objective is to thoroughly evaluate the zero-shot retrieval capabilities of several state-of-the-art sentence embedding models by meticulously comparing their performance across two distinct scenarios:

1.  **A complex, multi-hop reasoning benchmark dataset (HotpotQA):** Designed to test advanced information synthesis.
2.  **A large, single-hop, domain-specific dataset:** Automatically generated from real-world, structured university data (CSV-based).

This focused comparison aims to vividly highlight the inherent challenges and limitations faced by general-purpose embeddings when deployed in specialized knowledge domains without the benefit of explicit fine-tuning.

## Methodology 🧪

### Query Dataset Construction 📚

Two main query datasets were developed for comprehensive evaluation:

1.  **Benchmark Data (HotpotQA) 🌟**
    * **Source:** An adapted subset of the validation split from the HotpotQA distractor setting.
    * **Size:** Approximately 2,000 queries.
    * **Characteristics:** Specifically designed to challenge models with complex, multi-hop reasoning over general knowledge, often requiring the integration of information from several disparate sources.

2.  **Domain-Specific Data (CSV-based) 🎓**
    * **Source:** Programmatically generated from the structured metadata of **Hanbat National University's Computer Engineering faculty** (including professor details, research areas, and contact information) and the university's **complete course catalog** (listing course codes, titles, descriptions, and prerequisites).
    * **Size:** Comprises a substantial ~10,000 queries.
    * **Characteristics:** Primarily targets single-hop, exact-match retrieval. These queries were derived by converting internal CSV data directly into factual question-answer pairs (e.g., “What is the research field of Professor X?” or “What is the course code for Course Y?”). This approach simulates the typical informational needs within enterprise or institutional environments, presenting a stark contrast to the complexity of HotpotQA.
    * **Raw Data:** The raw, automatically generated domain-specific QA pairs can be found in `data/qa_course_dataset_en.csv` and `data/qa_professor_dataset_en.csv`.

### Embedding Models and Metrics 📊

We meticulously selected and evaluated a diverse range of embedding architectures renowned for their strong retrieval performance:

* **Selected Models:**
    * `MiniLM`
    * `MPNet` (serving as a robust, medium-sized architecture, representing TAS-B in this context)
    * `E5 family` (including `E5-small-v2`, `E5-base-v2`, `E5-large-v2`), which leverages advanced contrastive learning for superior embedding quality.
* **Evaluation Metrics:**
    * **Recall@5 ($R@5$):** Measures the proportion of queries for which the correct answer is successfully identified within the top 5 retrieved documents.
    * **Mean Reciprocal Rank ($MRR$):** Quantifies the average rank of the first relevant document, penalizing models that retrieve relevant items lower in the list.
    * **Inference Time per Query:** Assesses the computational efficiency of each model during the retrieval process.

## Experimental Results 📈

The comprehensive experimental findings unveil distinct and insightful performance patterns when comparing the two disparate datasets.

### HotpotQA Benchmark Performance 🏆

**Table 1: Retrieval Performance on HotpotQA Benchmark (N $\approx$ 2,000 Queries)**

| Model Name / Feature                         | Size (M) | Recall@5 | MRR    | Time/Query (ms) |
| :------------------------------------------- | :------- | :------- | :----- | :-------------- |
| MiniLM                                       | 22       | 0.7300   | 0.6308 | 2.95            |
| MPNet                                        | 110      | 0.7665   | 0.6621 | 24.90           |
| MultiMiniLM                                  | 38       | 0.4690   | 0.3663 | 3.32            |
| E5-small-v2                                  | 50       | 0.8620   | 0.7960 | 10.38           |
| E5-base-v2                                   | 110      | 0.8645   | 0.8038 | 19.98           |
| E5-large-v2 (Highest performance in E5 family) | 330      | 0.8750   | 0.8200 | 35.00           |

* **Key Observation:** E5 models consistently demonstrated superior performance, underscoring their robustness and effectiveness in handling complex, benchmark-driven retrieval challenges.
* **Raw Data:** Detailed results can be found in `data/hotpot_eval_2000_results-2.csv`.

### Domain-Specific Performance 🏢

**Table 2: Retrieval Performance on Domain-Specific Data (N $\approx$ 10,000 Queries)**

| Model Name / Feature                         | Size (M) | Recall@5 | MRR    | Time/Query (ms) |
| :------------------------------------------- | :------- | :------- | :----- | :----- |
| MiniLM                                       | 22       | 0.6441   | 0.5019 | 0.52   |
| MPNet                                        | 110      | 0.6116   | 0.4750 | 2.03   |
| MultiMiniLM                                  | 38       | 0.5211   | 0.3719 | 0.69   |
| E5-small-v2                                  | 50       | 0.6712   | 0.5335 | 0.70   |
| E5-base-v2                                   | 110      | 0.6468   | 0.4996 | 2.05   |
| E5-large-v2 (Highest performance in E5 family) | 330      | 0.6371   | 0.4931 | 7.83   |

* **Key Observation:** A notable decline in overall MRR and Recall@5 scores was observed across *all* models when evaluated on the larger, domain-specific dataset compared to HotpotQA. This outcome strongly suggests that the specialized nature of the questions or the unique characteristics of the automatically generated queries present a distinct and more challenging retrieval environment. Interestingly, E5-small-v2 maintained a strong relative position within this specialized domain.
* **Raw Data:** Detailed results are available in `data/model_eval_results.csv`.

**Visualizations 🖼️:**
For a deeper visual understanding, the `paper.pdf` includes:
* **Figure 1: Comparison of Recall@5 performance**, which graphically illustrates the consistent degradation in performance across all models when moving from general to domain-specific retrieval tasks.
* **Figure 2: Model Parameters vs. Recall@5 Performance**, a scatter plot demonstrating that while larger models generally achieve higher Recall@5, this advantage does not eliminate the substantial performance gap encountered in zero-shot domain-specific applications.

## Discussion and Conclusion 🤔💡

Our study's most striking finding is the **consistent and significant drop in Recall@5 for all tested models** when transitioning from the broad HotpotQA benchmark to the focused, specialized university domain. This emphatically highlights a critical inherent limitation of general-purpose embedding models: their constrained adaptability to highly specialized domains without the benefit of targeted additional training or fine-tuning. Even the more robust, larger models, despite their overall superior performance, could not fully bridge this substantial performance gap, indicating that raw scale is not a panacea for domain specificity.

**Key Takeaways from the Research:**
* **Domain Specificity Challenge:** Zero-shot embeddings demonstrably struggle with domain-bound vocabulary, the specificity of entities, and the unique structural variance of questions prevalent in specialized domains. These factors are often less pronounced in open-domain benchmarks.
* **Model Scale Insufficiency:** While larger models generally exhibit higher absolute Recall@5 scores, their increased size alone is insufficient to fully overcome the fundamental challenge of robust zero-shot domain transfer. The performance gap persists.

**Future Directions and Recommendations:**
To overcome these limitations and unlock the full potential of embedding models in real-world, specialized contexts, future research should urgently prioritize:
* **Domain-Adaptive Fine-Tuning:** Tailoring models with authentic, in-domain corpora.
* **Contrastive Representation Learning:** Developing techniques that enhance domain-specific distinctions.
* **Semantic Augmentation:** Enriching query and document representations with domain knowledge.
* **Adaptive Negative Sampling Strategies:** Improving the learning process by selecting more challenging negative examples relevant to the domain.

These focused efforts are paramount for developing truly robust and deployable agentic retrieval systems for enterprise and specialized knowledge domains. By advancing these low-resource domain adaptation and few-shot learning techniques, we can move beyond the current limitations of generic zero-shot capabilities, ensuring that AI agents can effectively and reliably ground information in specific, real-world contexts.

## Project Structure 📁

Our project's file organization is as follows:

```
.
├── Embedded.tex                       # 📄 LaTeX source for paper
├── paper.pdf                          # 📦 Compiled paper PDF
├── data/                              # 🗃️ Datasets & results
│   ├── hotpot\_eval\_2000\_results-2.csv # 📊 HotpotQA evaluation
│   ├── model\_eval\_results.csv         # 📈 Univ. domain evaluation
│   ├── qa\_course\_dataset\_en.csv       # 📚 Course QA data (EN)
│   └── qa\_professor\_dataset\_en.csv    # 🧑‍🏫 Professor QA data (EN)
└── src/                               # 💻 Python evaluation scripts
├── Embedding\_hotpotQA.py          # ✨ HotpotQA retrieval script
└── Embedding\_univ\_domain.py       # 🏫 Univ. domain retrieval script
```

## How to Reproduce 🛠️

Follow these steps to reproduce the paper's results and generate the PDF:

### 1. Compile the Paper 📝

To compile `Embedded.tex` into `paper.pdf`, you will need a LaTeX distribution (e.g., TeX Live, MiKTeX). For optimal results and font rendering (due to `fontspec`), please use **XeLaTeX** or **LuaLaTeX**.

```bash
# Navigate to the root of the repository
cd your-repo-name

# Compile the LaTeX document
# You might need to run this command several times (2-3 times)
# for all cross-references, table of contents, and bibliography to settle correctly.
xelatex Embedded.tex
xelatex Embedded.tex
xelatex Embedded.tex
```

### 2\. Run the Evaluation Scripts 💻

The Python scripts in the `src/` directory were used to perform the embedding model evaluations.

```bash
# Ensure you are in the root directory of the repository
cd your-repo-name

# 📦 Create and activate a virtual environment (highly recommended!)
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate

# ⬇️ Install required Python packages
# (Please ensure these are the exact dependencies of your scripts, 
# you might generate a requirements.txt using `pip freeze > requirements.txt` after installing them manually)
pip install sentence-transformers pandas numpy scikit-learn

# 🏃‍♀️ Run HotpotQA evaluation (Example command - adjust based on your script's arguments)
# If your raw HotpotQA data is named differently or located elsewhere, adjust the path.
# Assuming your script can take a dataset path as an argument.
python src/Embedding_hotpotQA.py --dataset_path data/qa_hotpotqa_dataset_en.csv 

# 🏃‍♂️ Run university domain evaluation (Example command - adjust based on your script's arguments)
# This assumes your script takes separate paths for course and professor data.
python src/Embedding_univ_domain.py \
    --course_data data/qa_course_dataset_en.csv \
    --professor_data data/qa_professor_dataset_en.csv
```

**Note on Script Arguments:** The exact command-line arguments for `Embedding_hotpotQA.py` and `Embedding_univ_domain.py` will depend on your specific implementation. Please refer to your scripts for the correct usage. If the raw HotpotQA dataset is not included in the `data/` directory, you may need to acquire it separately.

## Requirements ✅

* **LaTeX Distribution:** A modern LaTeX distribution (e.g., TeX Live, MiKTeX) capable of compiling with **XeLaTeX** or **LuaLaTeX**.
* **Python:** Version 3.8 or higher.
* **Python Libraries:**
    * `sentence-transformers`
    * `pandas`
    * `numpy`
    * `scikit-learn` (for computing metrics like MRR, Recall)

## License 📄

This project is open-sourced under the MIT License. See the [LICENSE](https://www.google.com/search?q=https://github.com/your-username/your-repo-name/blob/main/LICENSE) file for more details.

## Contact 📧

For any questions, feedback, or potential collaborations, please feel free to open an issue in this repository or reach out directly to Min-ho Hwang at ddoli9902@naver.com. I'd love to hear from you\!

-----
