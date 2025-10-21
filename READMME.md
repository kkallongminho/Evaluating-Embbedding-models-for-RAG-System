네, 요청하신 내용을 바탕으로 GitHub README.md 파일을 영어로 작성해 드리겠습니다. 프로젝트의 주요 정보와 파일 구조, 사용법 등을 포함하도록 구성했습니다. 코드 블록 형태로 제공해 드릴 테니, 복사하여 GitHub에 붙여넣으시면 됩니다.

```markdown
# Evaluating Domain-Specific Zero-Shot Retrieval without LLMs

This repository contains the LaTeX source code for the paper "Evaluating Domain-Specific Zero-Shot Retrieval without LLMs: A Comparative Study of CSV-Derived and Benchmark Question Generation," along with all experimental data and key Python scripts.

The study investigates the retrieval performance of embedding-based Question Answering (QA) systems in a zero-shot setting, without relying on large language models (LLMs). We compare performance on a standard academic benchmark (HotpotQA) against a dataset automatically generated from structured, real-world university domain data.

## Table of Contents

- [Evaluating Domain-Specific Zero-Shot Retrieval without LLMs](#evaluating-domain-specific-zero-shot-retrieval-without-llms)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Paper Abstract](#paper-abstract)
  - [Datasets](#datasets)
  - [Experimental Results](#experimental-results)
  - [Usage](#usage)
  - [Requirements](#requirements)
  - [License](#license)
  - [Contact](#contact)

## Project Structure

```

.
├── Embedded.tex                       \# LaTeX source code for the paper
├── paper.pdf                          \# Compiled PDF of the paper
├── data/
│   ├── hotpot\_eval\_2000\_results-2.csv \# Evaluation results for HotpotQA benchmark
│   ├── model\_eval\_results.csv         \# Evaluation results for the university domain dataset
│   ├── qa\_course\_dataset\_en.csv       \# QA dataset for university course information (English)
│   └── qa\_professor\_dataset\_en.csv    \# QA dataset for university professor information (English)
└── src/
├── Embedding\_hotpotQA.py          \# Python script for evaluating HotpotQA retrieval
└── Embedding\_univ\_domain.py       \# Python script for evaluating university domain retrieval

````

## Paper Abstract

In this study, we investigate the retrieval performance of embedding-based QA systems in a zero-shot setting without utilizing large language models. We construct two distinct query datasets: one automatically generated from structured domain data (CSV-based) and another adapted from the HotpotQA benchmark. Using various embedding architectures, including E5, MiniLM, and TAS-B (represented by MPNet in this experiment), we evaluate their effectiveness in domain-specific retrieval tasks. The study presents insights into how model architecture and question source impact retrieval accuracy, mean reciprocal rank, and recall. This work contributes to understanding lightweight, agentic retrieval systems that operate without generative reasoning components, aligning with the principles of agentic autonomy and information grounding.

**Keywords:** Zero-Shot Retrieval, Embedding Models, HotpotQA, Agentic Systems, Information Grounding.

## Datasets

The `data/` directory contains the following datasets:

* `qa_course_dataset_en.csv`: A dataset of question-answer pairs related to university course information.
* `qa_professor_dataset_en.csv`: A dataset of question-answer pairs related to university professor details.
* These two datasets form the **Domain-Specific Data** mentioned in the paper, comprising approximately 10,000 queries when combined and processed.
* **HotpotQA Data:** The paper refers to an adapted validation split of the HotpotQA distractor setting for benchmark data, consisting of approximately 2,000 queries. This specific raw dataset is not directly included due to its size and external origin, but the evaluation results are provided.

## Experimental Results

The results of our retrieval experiments are stored in the following CSV files within the `data/` directory:

* `hotpot_eval_2000_results-2.csv`: Contains the detailed evaluation metrics (Recall@5, MRR, Time/Query) for various embedding models on the HotpotQA benchmark.
* `model_eval_results.csv`: Contains the detailed evaluation metrics for the same embedding models on the automatically generated university domain-specific dataset.

These results are further analyzed and visualized in the `paper.pdf`.

## Usage

### 1. Recompile the Paper

To compile the LaTeX paper (`Embedded.tex`) into a PDF:

```bash
# Ensure you have a LaTeX distribution installed (e.g., TeX Live, MiKTeX)
# For best results with fontspec, use XeLaTeX or LuaLaTeX
xelatex Embedded.tex
# You might need to run it multiple times for references and TOC to settle
xelatex Embedded.tex
xelatex Embedded.tex
````

### 2\. Run the Evaluation Scripts

The `src/` directory contains Python scripts used for evaluating the embedding models.

**Note:** These scripts require specific libraries (e.g., `sentence-transformers`, `pandas`, `scikit-learn`). It is highly recommended to set up a virtual environment.

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Create a virtual environment
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate

# Install necessary Python packages
pip install -r requirements.txt # (assuming you create this file based on scripts' dependencies)

# To run the HotpotQA evaluation (example, specific arguments might be needed)
python src/Embedding_hotpotQA.py

# To run the university domain evaluation (example, specific arguments might be needed)
python src/Embedding_univ_domain.py
```

  * **Note on `requirements.txt`:** Please generate a `requirements.txt` file based on the actual dependencies of `Embedding_hotpotQA.py` and `Embedding_univ_domain.py` (e.g., `sentence-transformers`, `pandas`, `numpy`). You can typically do this with `pip freeze > requirements.txt` after installing dependencies.

## Requirements

  * A LaTeX distribution (e.g., TeX Live, MiKTeX) with XeLaTeX or LuaLaTeX compiler.
  * Python 3.8+
  * The following Python libraries (example, please verify from scripts):
      * `sentence-transformers`
      * `pandas`
      * `numpy`
      * `scikit-learn` (for metrics like MRR, Recall)

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details. (If you have a specific license, update this accordingly. Otherwise, you might need to create a `LICENSE` file.)

## Contact

For any questions or inquiries, please open an issue in this repository or contact Min-ho Hwang at [your.email@example.com](mailto:your.email@example.com).

```
```
