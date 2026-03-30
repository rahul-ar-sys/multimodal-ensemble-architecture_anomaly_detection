# Bottle Stacked Ensemble: "Boosted-Bagging" Architecture

## Overview
A high-fidelity ensemble anomaly detection pipeline designed without the constraints of real-time conveyor speed. This system prioritizes **Maximal Accuracy** and **Deep Feature Extraction** by employing a "Boosted-Bagging" approach. 

The strategy utilizes pre-existing datasets (`RGB`, `IR`, `Concat`) to process each stream independently. Each stream acts as a specialized auditor through a sequential error-correction chain, and their outputs are ultimately funneled into a machine-learning tree-based stacker (The Umpire).

---

## 🏗 System Architecture

The following diagram outlines the data flow across the **parallel bags** (Level-0) and how they merge into the final **meta-learner** (Level-1).

```mermaid
graph TD
    subgraph Data_Bagging_Layer [Data Bagging Layer: Multimodal Inputs]
        D1[RGB Dataset]
        D2[IR Dataset]
        D3[Concat Dataset]
    end

    subgraph Boosted_Chain_RGB [Bag 1: RGB Expert Chain]
        direction TB
        PC_RGB[PatchCore: Base Expert] --> SN_RGB[SimpleNet: Residual Refiner]
        SN_RGB --> Score_RGB[RGB Anomaly Vector]
    end

    subgraph Boosted_Chain_IR [Bag 2: IR Expert Chain]
        direction TB
        PC_IR[PatchCore: Base Expert] --> SN_IR[SimpleNet: Residual Refiner]
        SN_IR --> Score_IR[IR Anomaly Vector]
    end

    subgraph Boosted_Chain_Concat [Bag 3: Concat Expert Chain]
        direction TB
        PC_CC[PatchCore: Base Expert] --> SN_CC[SimpleNet: Residual Refiner]
        SN_CC --> Score_CC[Concat Anomaly Vector]
    end

    %% Connections from Data to Bags
    D1 --> Boosted_Chain_RGB
    D2 --> Boosted_Chain_IR
    D3 --> Boosted_Chain_Concat

    subgraph Stacking_Layer [Level-1: Meta-Learner]
        direction TB
        Vector[6-Dimensional Feature Vector]
        ML[Meta-Learner: XGBoost / Random Forest]
        Vector --> ML
    end

    %% Connections to Stacking
    Score_RGB --> Vector
    Score_IR --> Vector
    Score_CC --> Vector

    subgraph Output_Layer [Final Decision]
        Result{Defect / Normal}
        Heatmap[Fused Anomaly Localization Map]
    end

    ML --> Result
    ML --> Heatmap

    %% Styling
    style Data_Bagging_Layer fill:#f9f,stroke:#333,stroke-width:2px
    style Stacking_Layer fill:#bbf,stroke:#333,stroke-width:2px
    style Output_Layer fill:#bfb,stroke:#333,stroke-width:2px
```

---

## 📂 Production Directory Structure

The structure physically isolates our parallel Datasets-to-Stream mapping while enforcing a modular engine for the boosting mechanics.

```text
bottle_stacked_ensemble/
├── data/                  # Pre-existing datasets
│   ├── rgb/
│   ├── ir/
│   └── concat/
├── src/
│   ├── models/            # Level-0 Entities
│   │   ├── patchcore.py   # Base Expert (Cold Anomalies)
│   │   └── simplenet.py   # Refiner Expert (Texture/Noise Anomalies)
│   ├── engine/            # Training Logic
│   │   ├── booster.py     # Mechanics to train SN on PC's residuals
│   │   └── bagger.py      # Manages the parallel datastreams (RGB vs IR vs Concat)
│   └── stacker.py         # Level-1 Meta-Learner
├── checkpoints/           # Serialization output for weights
│   ├── rgb_chain/
│   ├── ir_chain/
│   └── concat_chain/
└── evaluate_ensemble.py   # Final AUROC/PRO metric scripts
```

---

## ⚙️ Technical Execution Details

### Phase 1: Level-0 "Expert" Training (The Bags)
Inside each stream (RGB, IR, Concat), the pipeline is run synchronously due to the requirement of "Boosting" within the bag:

1. **The Base Layer (PatchCore):** Trained first to generate a spatial coreset memory bank of "Cold" structures.
2. **Identifying Residuals:** Upon internal testing of the training set, we extract anomaly maps that produce "False Positives" (e.g. normal bottles that trigger high variance due to texture noise).
3. **The Boosting Layer (SimpleNet):** We pass these specific hard samples directly to SimpleNet. SimpleNet specializes perfectly in the high-frequency/texture anomalies that evaded the coreset.

### Phase 2: Feature Stacking (Level-1)
Once the 6 distinct models are trained (2 models × 3 modality streams):
1. **Out-of-Fold (OOF) Inference:** We run 5-fold cross-validation on the validation/test targets.
2. **Feature Extraction:** Every bottle maps to a unified 6-dimensional predictive vector:
   `V = [Score_RGB_PC, Score_RGB_SN, Score_IR_PC, Score_IR_SN, Score_Concat_PC, Score_Concat_SN]`
3. **XGBoost Meta-Learner:** A downstream Random Forest/XGBoost model is trained purely on these 6-D vectors, acting as logic gates. For example: *"If IR says defect but RGB says normal, it is likely an internal crack—mark as Defect."*

### ⚠️ Constraints & Optimizations 
- **Backbone Density for Concat:** Because the `concat` stream aggregates physical layer dimensions, we widen the backbone of the `concat` base expert (e.g., using `WideResNet-101`) compared to the independent `RGB` / `IR` pipelines.
- **Hardware Optimization:** To prevent VRAM crashes during stacking—since holding three independent PatchCore coreset memory banks concurrently is destructive—we Pickle the Level-0 score tensors to the CPU. The XGBoost learner processes these pre-calculated static tensors natively.
- **Metrics Standard:** Models prioritize localization fidelity; optimizations are tuned tightly against **Pixel-level AUROC** and **PRO (Per-Region Overlap)** over raw image accuracy.
