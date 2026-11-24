# Cross-Lingual Prediction of Engagement from Non-Verbal Behaviour (NoXi)

Student project (Deep Learning), Shubham Kumar, 23CSU388, CSE, The NorthCap University. Supervisor: Prof. Poonam Chaudhary.

**Question.** Can a computer estimate, every second, how engaged a person is in a conversation using only the voice and the face (not the words)? How well does this work for people it has never seen, what is lost when the language changes (English, French, German), and which signal, voice or face, carries over better?

## Data
[NoXi](https://multimediate.perceptualui.org/datasets/Dataset_NoXi/) expert-novice conversations. This project uses **24 conversations** (8 English, 8 French, 8 German): 48 recordings and 48,486 one-second windows. The engagement label runs from 0 to 1.

**The NoXi data is licensed (EULA) and is NOT included in this repository.** It must be requested from its owners. All results here are stored as tables and charts, so the result notebooks run without the data.

## Inputs
- eGeMAPS voice features (88)
- OpenFace face features (709)
- w2v-BERT 2.0 speech embedding (1,024, frozen, pretrained)
- VideoMAE video embedding (1,408, frozen, pretrained)
- Context: the mean of the speaker's own features over the last 5 s and the last 10 s (past only)

## Testing design
- Speaker-independent 6-fold cross-validation (no person in both training and test)
- Leave-one-language-out: train on two languages, test on the third
- Within vs cross language with the same training size
- A held-out test set (6 sessions) used once
- 5 seeds for the neural networks; metric CCC; bootstrap confidence intervals, Wilcoxon tests, Holm correction

## Main results (CCC, higher is better, speaker-independent)
| Model | CCC |
|---|---|
| Linear regression | 0.476 |
| XGBoost | 0.625 |
| Random forest | 0.628 |
| ANN (797-512-256-1) | 0.569 |
| Transformer trained from scratch (last 10 s) | 0.654 |
| Dense head on w2v-BERT 2.0 + VideoMAE | 0.676 |
| **Dense head on w2v-BERT 2.0 + VideoMAE + context** | **0.767 +- 0.017** (0.733 on a new language) |

Other findings: a random split overstates the score (about 0.89 against 0.63); voice (0.654) is much stronger than face (0.361); a new language costs only about 0.04 CCC; past context gives the biggest gain. Limits: 24 of 72 conversations, language and recording site cannot be separated, and a role-only baseline already reaches 0.434.

## Repository layout
| Folder / file | Content |
|---|---|
| `notebooks/` | 19 notebooks (each starts with an "In short" cell). 01-13 are the development steps; **18, 28-35 are the final results and only load saved tables** |
| `src/` | the scripts behind the notebooks; final models in `29_final_models.py`, extra analyses in `30`-`35` |
| `tools/data24/` | scripts that built the 24-conversation feature tables from the raw NoXi files (needs the data) |
| `reports/` | all result tables (`*.csv`) and charts |
| `charts/` | saved pictures used by notebooks 33 and 35 |
| `run.py` | runs notebooks by number |
| `HOW_TO_RUN_VSCODE.md` | step-by-step running instructions |

Key notebooks: **28** review experiments (Gradient Boosting, all protocols), **29** the seven final models, **30** causal smoothing, **31** language probe, **32** speed and size, **33** SHAP, **34** expert vs novice, **35** error analysis.

## Run
```
pip install -r requirements.txt
python run.py 18 28 29 30 31 32 33 34 35
```
(Python 3.12; TensorFlow 2.21, scikit-learn 1.4, XGBoost 3.4.) Notebooks 18 and 28-35 need no NoXi data.

## Licence
Code under the MIT licence (see `LICENSE`). The NoXi data and its embeddings are not part of this repository.
