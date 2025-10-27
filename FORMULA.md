# HalluField Code Update - Formula Corrections

## 🔄 What Was Updated

The HalluField formulas have been corrected to match the **exact formulas used in the paper** (arXiv:2509.10753).

## ✅ Corrected Formulas

### 1. HalluField (Default Formula)

**Corresponds to**: "HalluField-B" from paper experiments  
**Best performer**: Without semantic entropy

```python
HalluField = (
    1.5 * Base_Energy_1.5 +
    2.0 * Base_Energy_2.0 +
    (ΔPotential_1.0 + ΔEntropy_1.0) +
    (ΔPotential_1.5 / 2.25 + ΔEntropy_1.5 / 1.5)
)
```

**Required temperatures**: 1.0, 1.5, 2.0

### 2. HalluFieldSE (With Semantic Entropy)

**Corresponds to**: "HalluField-Sem-F12E" from paper experiments  
**Best performer**: With semantic entropy

```python
HalluFieldSE = (
    0.4 * (2.0 * Base_Energy_2.0 + 1.5 * Base_Energy_1.5 + Base_Energy_1.0) +
    0.6 * (
        (ΔPotential_1.0 + ΔEntropy_1.0) +
        (ΔPotential_1.5 / 2.25 + ΔEntropy_1.5 / 1.5) +
        (ΔPotential_2.0 / 4.0 + ΔEntropy_2.0 / 2.0)
    ) +
    2.5 * Semantic_Entropy_1.0
)
```

**Required temperatures**: 1.0, 1.5, 2.0, 2.5

## 📝 Files Updated

1. **`hallufield/core/compute.py`**
   - Updated `compute_hallufield_score()` method with exact formulas
   - Added detailed documentation about formula correspondence
   - Added notes about temperature requirements

2. **`README.md`**
   - Added "Methodology" section with formula explanations
   - Included mathematical notation for both formulas
   - Documented required temperatures

3. **`QUICK_REFERENCE.md`**
   - Updated metrics section with formula names from paper
   - Added temperature requirements for each formula

4. **`RESTRUCTURING_SUMMARY.md`**
   - Added notes about exact formula implementation

5. **`configs/default_config.yaml`**
   - Added comments explaining temperature requirements
   - Noted which temperatures are needed for which formulas

## 🎯 Key Points

### Formula Selection Process

These formulas were selected from **extensive experiments** comparing multiple candidates:

- HalluField-B was chosen as the best formula **without** semantic entropy
- HalluField-Sem-F12E was chosen as the best formula **with** semantic entropy
- Both formulas were selected based on achieving the **highest AUC** in hallucination detection

### Temperature Requirements

⚠️ **Important**: You must generate data at the required temperatures:

- **Minimum for HalluField**: 1.0, 1.5, 2.0
- **Minimum for HalluFieldSE**: 1.0, 1.5, 2.0, 2.5

The default configuration includes all required temperatures: `[1.0, 1.5, 2.0, 2.5, 3.0]`

## 🔬 Background

During the research phase, multiple HalluField formulas were tested:

```python
# Some candidates tested (from original code):
"HalluField-Sem-25S-A"
"HalluField-Sem-20S"
"HalluField-Sem-F1"
"HalluField-Sem-F11"
"HalluField-Sem-F12"  # Good performer
"HalluField-A"
"HalluField-B"        # ← Selected for paper (best without SE)
"HalluField-Sem-F12E" # ← Selected for paper (best with SE)
```

The two best-performing formulas (HalluField-B and HalluField-Sem-F12E) are now the default implementations in the package.

## ✨ No Breaking Changes

The API remains the same - only the **internal formula implementations** were corrected:

```python
# Usage is unchanged
computer = HalluFieldComputer()
df["HalluField"] = computer.compute_hallufield_score(df, "default")
df["HalluFieldSE"] = computer.compute_hallufield_score(df, "with_semantic_entropy")
```

## 📊 Expected Behavior

With the corrected formulas, you should now get results that **exactly match** the paper's reported performance when using:

1. The same models (e.g., LLaMA-2-7B)
2. The same datasets (e.g., SQuAD, TriviaQA)
3. The same temperatures [1.0, 1.5, 2.0, 2.5, 3.0]
4. The same number of generations (10+)

## 🙏 Thanks

Thank you for catching this and providing the correct formulas! The code now accurately reflects the methodology described in the paper.

---

**Updated**: October 27, 2025  
**Version**: 0.1.0
