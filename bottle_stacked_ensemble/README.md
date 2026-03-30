# Bottle Stacked Ensemble: "Boosted-Bagging" Architecture

## Overview
A high-fidelity ensemble anomaly detection pipeline designed without the constraints of real-time conveyor speed. This system prioritizes **Maximal Accuracy** and **Deep Feature Extraction** by employing a "Boosted-Bagging" approach. 

The strategy utilizes pre-existing datasets (`RGB`, `IR`, `Concat`) to process each stream independently. Each stream acts as a specialized auditor through a sequential error-correction chain, and their outputs are ultimately funneled into a machine-learning tree-based stacker (The Umpire).

---

## 🏗 System Architecture

The following diagram outlines the data flow across the **parallel bags** (Level-0) and how they merge into the final **meta-learner** (Level-1).

```mermaid
graph TD
    classDef base fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef refiner fill:#065f46,stroke:#10b981,stroke-width:2px,color:#fff;
    classDef union fill:#4c1d95,stroke:#8b5cf6,stroke-width:2px,color:#fff;
    classDef final fill:#991b1b,stroke:#ef4444,stroke-width:2px,color:#fff;

    subgraph Bagging Level 0: Parallel Specialized Streams
        direction TB
        
        subgraph RGB Stream
            R_Data[(RGB Dataset)] -->|Train| R_PC[PatchCore Base]:::base
            R_PC -->|Residuals / Hard Samples| R_SN[SimpleNet Refiner]:::refiner
            R_Data -->|Inference| R_PC
            R_Data -->|Inference| R_SN
            R_PC -.-> R_Score(Score RGB_PC)
            R_SN -.-> R_Score_SN(Score RGB_SN)
        end

        subgraph IR Stream
            I_Data[(IR Dataset)] -->|Train| I_PC[PatchCore Base]:::base
            I_PC -->|Residuals / Hard Samples| I_SN[SimpleNet Refiner]:::refiner
            I_Data -->|Inference| I_PC
            I_Data -->|Inference| I_SN
            I_PC -.-> I_Score(Score IR_PC)
            I_SN -.-> I_Score_SN(Score IR_SN)
        end

        subgraph Concat Stream
            C_Data[(Concat Dataset)] -->|Train| C_PC[PatchCore Base<br>WideResNet-101]:::base
            C_PC -->|Residuals / Hard Samples| C_SN[SimpleNet Refiner]:::refiner
            C_Data -->|Inference| C_PC
            C_Data -->|Inference| C_SN
            C_PC -.-> C_Score(Score Concat_PC)
            C_SN -.-> C_Score_SN(Score Concat_SN)
        end
    end

    subgraph Stacking Level 1: Meta-Learner (The Umpire)
        OOF[OOF 6-Dimensional Vector<br>v = R_PC, R_SN, I_PC, I_SN, C_PC, C_SN]:::union
        R_Score --> OOF
        R_Score_SN --> OOF
        I_Score --> OOF
        I_Score_SN --> OOF
        C_Score --> OOF
        C_Score_SN --> OOF
        
        OOF -->|Fit| Umpire{Level-1 Stacker<br/>XGBoost / Random Forest}:::final
        Umpire --> Final[Final Decision<br>Pixel AUROC / PRO Score]
    end
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
