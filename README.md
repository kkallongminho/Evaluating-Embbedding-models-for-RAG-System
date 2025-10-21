# Evaluating Domain-Specific Zero-Shot Retrieval without LLMs

This repository presents the research detailed in the paper "Evaluating Domain-Specific Zero-Shot Retrieval without LLMs: A Comparative Study of CSV-Derived and Benchmark Question Generation." It includes the full LaTeX source, compiled PDF, experimental datasets, and key Python scripts used for model evaluation.

## Table of Contents

- [Evaluating Domain-Specific Zero-Shot Retrieval without LLMs](#evaluating-domain-specific-zero-shot-retrieval-without-llms)
  - [Project Overview](#project-overview)
  - [Research Goal](#research-goal)
  - [Methodology](#methodology)
    - [Query Dataset Construction](#query-dataset-construction)
    - [Embedding Models and Metrics](#embedding-models-and-metrics)
  - [Experimental Results](#experimental-results)
    - [HotpotQA Benchmark Performance](#hotpotqa-benchmark-performance)
    - [Domain-Specific Performance](#domain-specific-performance)
  - [Discussion and Conclusion](#discussion-and-conclusion)
  - [Project Structure](#project-structure)
  - [How to Reproduce](#how-to-reproduce)
  - [Requirements](#requirements)
  - [License](#license)
  - [Contact](#contact)

## Project Overview

The proliferation of Large Language Models (LLMs) has transformed information retrieval, but their computational demands can be substantial. This research focuses on **lightweight, embedding-based retrieval systems operating in a zero-shot setting**, crucial for efficient and grounded agentic systems. Our core objective is to understand how different types of evaluation data—standard benchmarks versus real-world, domain-specific data—influence the measured performance of these embedding models.

## Research Goal

To evaluate the zero-shot retrieval capabilities of various sentence embedding models by comparing their performance on:
1.  A complex, multi-hop reasoning benchmark dataset (HotpotQA).
2.  A large, single-hop, domain-specific dataset automatically generated from structured university information (CSV-based).
This comparison aims to highlight the challenges and limitations of general-purpose embeddings in specialized knowledge domains without explicit fine-tuning.

## Methodology

### Query Dataset Construction

Two primary query datasets were utilized for evaluation:

1.  **Benchmark Data (HotpotQA):**
    * **Source:** Adapted from the validation split of the HotpotQA distractor setting.
    * **Size:** Approximately 2,000 queries.
    * **Characteristics:** Tests complex, multi-hop reasoning over general knowledge, requiring models to integrate information from multiple sources.

2.  **Domain-Specific Data (CSV-based):**
    * **Source:** Programmatically generated from the structured metadata of **Hanbat National University's Computer Engineering faculty** (professor details, research areas, contact info) and the university's **entire course catalog** (course codes, titles, descriptions, prerequisites).
    * **Size:** Comprising approximately 10,000 queries.
    * **Characteristics:** Primarily tests single-hop, exact-match retrieval, derived from converting internal CSV data into question-answer pairs (e.g., “What is the research field of Professor X?”). This represents typical informational needs in enterprise or institutional domains, contrasting sharply with HotpotQA's complexity.
    * **Raw Data:** The `data/qa_course_dataset_en.csv` and `data/qa_professor_dataset_en.csv` files contain the raw, automatically generated domain-specific QA pairs.

### Embedding Models and Metrics

* **Selected Models:** We evaluated a variety of high-performing embedding architectures:
    * MiniLM
    * MPNet (representing TAS-B in this experiment)
    * E5 family (E5-small-v2, E5-base-v2, E5-large-v2), known for contrastive learning.
* **Evaluation Metrics:**
    * **Recall@5 ($R@5$):** The proportion of queries for which the correct answer is found within the top 5 retrieved documents.
    * **Mean Reciprocal Rank ($MRR$):** A measure of the average rank of the first relevant document.
    * **Inference Time per Query:** To assess computational efficiency.

## Experimental Results

The experimental findings reveal distinct performance patterns across the two datasets.

### HotpotQA Benchmark Performance

**Table 1: Retrieval Performance on HotpotQA Benchmark (N $\approx$ 2,000 Queries)**

| Model Name / Feature                         | Size (M) | Recall@5 | MRR    | Time/Query (ms) |
| :------------------------------------------- | :------- | :------- | :----- | :-------------- |
| MiniLM                                       | 22       | 0.7300   | 0.6308 | 2.95            |
| MPNet                                        | 110      | 0.7665   | 0.6621 | 24.90           |
| MultiMiniLM                                  | 38       | 0.4690   | 0.3663 | 3.32            |
| E5-small-v2                                  | 50       | 0.8620   | 0.7960 | 10.38           |
| E5-base-v2                                   | 110      | 0.8645   | 0.8038 | 19.98           |
| E5-large-v2 (Highest performance in E5 family) | 330      | 0.8750   | 0.8200 | 35.00           |

* **Observation:** E5 models consistently outperformed others, demonstrating strong robustness in complex, benchmark-driven retrieval tasks.
* **Raw Data:** See `data/hotpot_eval_2000_results-2.csv`.

### Domain-Specific Performance

**Table 2: Retrieval Performance on Domain-Specific Data (N $\approx$ 10,000 Queries)**

| Model Name / Feature                         | Size (M) | Recall@5 | MRR    | Time/Query (ms) |
| :------------------------------------------- | :------- | :------- | :----- | :-------------- |
| MiniLM                                       | 22       | 0.6441   | 0.5019 | 0.52            |
| MPNet                                        | 110      | 0.6116   | 0.4750 | 2.03            |
| MultiMiniLM                                  | 38       | 0.5211   | 0.3719 | 0.69            |
| E5-small-v2                                  | 50       | 0.6712   | 0.5335 | 0.70            |
| E5-base-v2                                   | 110      | 0.6468   | 0.4996 | 2.05            |
| E5-large-v2 (Highest performance in E5 family) | 330      | 0.6371   | 0.4931 | 7.83            |

* **Observation:** Overall MRR and Recall@5 scores were significantly lower across all models compared to HotpotQA. This suggests that the domain-specific nature of questions or the characteristics of automatically generated queries present a unique challenge for zero-shot retrieval. E5-small-v2 maintained a relatively strong position within this domain.
* **Raw Data:** See `data/model_eval_results.csv`.

**Visualizations:**
The paper (`paper.pdf`) includes detailed figures, such as:
* **Figure 1: Comparison of Recall@5 performance** across models on both datasets, illustrating a consistent performance degradation in domain-specific tasks.
* **Figure 2: Model Parameters vs. Recall@5 Performance**, showing that while larger models generally perform better, they still suffer from a significant performance gap when applied to domain-specific data in a zero-shot manner.

## Discussion and Conclusion

The study's findings reveal a **consistent and significant drop in Recall@5 for all models** when transitioning from general HotpotQA data to the specialized university domain. This underscores a critical limitation of general-purpose embedding models: their limited adaptability to specialized domains without additional training or fine-tuning. Even larger models, while generally performing better, could not fully bridge this performance gap.

**Key Takeaways:**
* Zero-shot embeddings struggle with domain-bound vocabulary, entity specificity, and structural question variance in specialized domains.
* Increased model scale alone is insufficient to overcome the fundamental challenge of zero-shot domain transfer.

**Future Directions:**
Future research should focus on:
* **Domain-adaptive fine-tuning** and **contrastive representation learning** using authentic domain corpora.
* **Semantic augmentation** and **adaptive negative sampling** strategies.
These efforts are crucial for developing robust, deployable agentic retrieval systems for enterprise and specialized knowledge domains, moving beyond the limitations of generic zero-shot capabilities.

## Project Structure

. ├── Embedded.tex # LaTeX source code for the paper ├── paper.pdf # Compiled PDF of the paper ├── data/ │ ├── hotpot_eval_2000_results-2.csv # Evaluation results for HotpotQA benchmark │ ├── model_eval_results.csv # Evaluation results for the university domain dataset │ ├── qa_course_dataset_en.csv # QA dataset for university course information (English) │ └── qa_professor_dataset_en.csv # QA dataset for university professor information (English) └── src/ ├── Embedding_hotpotQA.py # Python script for evaluating HotpotQA retrieval └── Embedding_univ_domain.py # Python script for evaluating university domain retrieval


## How to Reproduce

### 1. Compile the Paper

To generate `paper.pdf` from `Embedded.tex`:

```bash
# Ensure you have a LaTeX distribution installed (e.g., TeX Live, MiKTeX)
# For best results with fontspec, use XeLaTeX or LuaLaTeX
xelatex Embedded.tex
# Run multiple times to resolve cross-references and table of contents
xelatex Embedded.tex
xelatex Embedded.tex
2. Run the Evaluation Scripts

The Python scripts in src/ are used for conducting the experiments.

Bash
# Clone the repository
git clone [YOUR_REPOSITORY_URL]
cd [YOUR_REPOSITORY_NAME]

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate

# Install required Python packages
# (Please ensure these are the correct dependencies from your scripts)
pip install sentence-transformers pandas numpy scikit-learn

# Example: Run HotpotQA evaluation
# (Specific command-line arguments or configurations might be needed based on your script)
python src/Embedding_hotpotQA.py --dataset_path data/qa_hotpotqa_dataset_en.csv # Example path if HotpotQA raw data was included

# Example: Run university domain evaluation
python src/Embedding_univ_domain.py \
    --course_data data/qa_course_dataset_en.csv \
    --professor_data data/qa_professor_dataset_en.csv
Note: The exact arguments for Embedding_hotpotQA.py and Embedding_univ_domain.py may vary based on how you implemented them. Please adjust the commands above to match your scripts' requirements. If your HotpotQA raw data (qa_hotpotqa_dataset_en.csv) is not in data/, you will need to acquire it separately or adapt the script.

Requirements
A LaTeX distribution (e.g., TeX Live, MiKTeX) with XeLaTeX or LuaLaTeX compiler.

Python 3.8+

Python libraries: sentence-transformers, pandas, numpy, scikit-learn.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any questions, suggestions, or collaborations, please open an issue in this repository or contact Min-ho Hwang at ddoli9902@naver.com.
