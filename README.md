# 🩺 Diabetic Retinopathy Lesion Classification using Machine Learning

Early detection of diabetic retinopathy (DR) is vital to prevent vision loss, but manual grading of retinal fundus images is time-consuming and difficult to scale.  
This project develops and evaluates a **lightweight, two-stage machine learning pipeline** for lesion classification and severity grading, implemented on the benchmark **DIARETDB1** dataset.

---

## 📖 Abstract

- **Problem:** Manual grading of DR images requires expert ophthalmologists and is not feasible for mass screening.  
- **Solution:** A modular pipeline with **preprocessing → segmentation → lesion detection → hierarchical classification → severity grading**.  
- **Models tested:** Logistic Regression, Random Forest, XGBoost.  
- **Results:**  
  - Bright lesions easier to detect than red lesions.  
  - Logistic Regression achieved the **highest lesion sensitivity** (≈0.85, ROC-AUC ≈0.88).  
  - XGBoost provided the **most balanced performance** for red lesions (Accuracy ≈0.71, ROC-AUC ≈0.81).  
  - Sub-type classification:  
    - **EX vs CWS** – LR best (Balanced Accuracy ≈0.83, ROC-AUC ≈0.92).  
    - **MA vs HM** – hardest task, LR best balance (Balanced Accuracy ≈0.62, ROC-AUC ≈0.68).  

---

## 🎯 Objectives

1. Enhance low-contrast retinal images through advanced preprocessing.  
2. Segment background structures (optic disc, vessels) to reduce misclassification.  
3. Detect and classify retinal lesions with hybrid feature extraction (appearance, structure, anatomy).  
4. Implement hierarchical classification for DR severity grading.  
5. Validate performance on **DIARETDB1** using Accuracy, Sensitivity, Specificity, ROC-AUC, PR-AUC.  

---

## 📊 Dataset

- **Name:** [DIARETDB1](https://www.it.lut.fi/project/imageret/diaretdb1/)  
- **Size:** 89 color fundus images (84 DR, 5 normal)  
- **Annotations:** Expert-validated for four lesion types:  
  - MA (Microaneurysms)  
  - HE (Haemorrhages)  
  - EX (Hard Exudates)  
  - CWS (Cotton Wool Spots)  

---

## ⚙️ Methodology / Pipeline

1. **Preprocessing:** Resize → Green channel extraction → CLAHE → Median filter → Morphological opening → Normalization.  
2. **Segmentation:** Vessel suppression & optic disc masking to remove normal anatomy.  
3. **Lesion Candidate Generation:** Sliding windows + hybrid feature extraction (30-D vector: intensity, structural, anatomical context).  
4. **Hierarchical Classification:**  
   - **Stage 1:** Lesion vs Background (Bright / Red families).  
   - **Stage 2:** Subtype classification → EX vs CWS, MA vs HM.  
5. **Severity Grading:** Aggregate lesion counts → classify as **No DR, Mild, Moderate, Severe**.  

---

## 🛠️ Technologies Used

- **Language:** Python  
- **Core Scientific:** NumPy, Pandas, SciPy  
- **Image Processing:** OpenCV, scikit-image  
- **Machine Learning:** scikit-learn (Logistic Regression, Random Forest, XGBoost)  
- **Visualization:** Matplotlib, Seaborn  
- **Utilities:** tqdm (progress bars), XML parsing for annotations  

---

## 📈 Results

### Stage 1: Lesion vs Background
| Model              | Task               | Sensitivity | Specificity | Accuracy | ROC-AUC |
|--------------------|-------------------|-------------|-------------|----------|---------|
| Logistic Regression | Bright vs Background | **0.85** | 0.71 | 0.78 | **0.88** |
| XGBoost            | Bright vs Background | 0.62 | 0.87 | 0.75 | 0.86 |
| Random Forest      | Bright vs Background | 0.46 | **0.92** | 0.70 | 0.84 |
| XGBoost            | Red vs Background   | 0.67 | 0.75 | **0.71** | **0.81** |

### Stage 2: Subtype Classification
- **EX vs CWS:** Logistic Regression best – Balanced Accuracy ≈ **0.83**, ROC-AUC ≈ **0.92**.  
- **MA vs HM:** Most challenging – Logistic Regression best balance with Balanced Accuracy ≈ **0.62**, ROC-AUC ≈ **0.68**.  

---

## 🚀 Installation & Usage

```bash
# Clone repository
git clone https://github.com/yourusername/diabetic-retinopathy-ml.git
cd diabetic-retinopathy-ml

# Install dependencies
pip install -r requirements.txt

# Train models
python main.py --mode train

# Evaluate models
python main.py --mode test
```

---

## ✅ Strengths

- Leakage-safe evaluation (grouped by image).  
- Transparent handling of class imbalance.  
- Modular, interpretable pipeline.  
- Strong results with lightweight models (no GPU required).  

---

## ⚠️ Limitations

- Evaluated only on DIARETDB1 (no external validation).  
- Limited hyperparameter tuning.  
- Stage-2 performance depends on Stage-1 accuracy (error propagation).  
- Class imbalance remains challenging for **rare lesions** (MA, CWS).  

---

## 🔮 Future Work

- Validate on larger, multi-institution datasets.  
- Explore CNN/deep learning for end-to-end lesion detection.  
- Improve detection of rare lesions with class-weighted methods or augmentation.  
- Extend system to real-world clinical workflows.  

---

## 📚 References

- Dissertation: *Diabetic Retinopathy Lesion Classification using Machine Learning*, Sibin Shibu, MSc Data Science, University of Roehampton, 2025.  
- DIARETDB1 Dataset: Kauppi et al., “DIARETDB1 diabetic retinopathy database and evaluation protocol,” 2007.  
- Additional references from dissertation (available in full report).  
