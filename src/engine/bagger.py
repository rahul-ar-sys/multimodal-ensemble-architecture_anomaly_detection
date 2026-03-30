"""
Bagger Engine

Manages parallel dataset streams (RGB, IR, Concat).
Extracts Level-0 Out-of-Fold vectors to disk to spare VRAM limits.
"""
import os
import pickle
import numpy as np

class Bagger:
    def __init__(self, booster_rgb, booster_ir, booster_concat):
        self.boosters = {
            'rgb': booster_rgb,
            'ir': booster_ir,
            'concat': booster_concat
        }

    def train_all_streams(self, data_loaders):
        """
        Iterates and builds the chain per parallel modality stream.
        data_loaders: Dict grouping loaders by stream, e.g., {'rgb': loader, etc.}
        """
        for stream_name, booster in self.boosters.items():
            print(f"--- Initiating parallel chain mapping for stream: {stream_name.upper()} ---")
            loader = data_loaders.get(stream_name)
            if loader:
                booster.train_chain(loader)
            else:
                print(f"Warning: No valid dataloader provided for {stream_name}.")
                
    def extract_oof_features(self, validation_loaders, save_dir="data"):
        """
        Cross-Validation phase. Evaluates logic strings dynamically across chains.
        V = [Score_{RGB_PC}, Score_{RGB_SN}, Score_{IR_PC}, Score_{IR_SN}, Score_{Concat_PC}, Score_{Concat_SN}]
        """
        print("--- Extracting Level-0 OOF Validation Ensembles ---")
        os.makedirs(save_dir, exist_ok=True)
        
        oof_vectors = []
        labels = []
        
        rgb_loader = validation_loaders.get('rgb', [])
        ir_loader = validation_loaders.get('ir', [])
        concat_loader = validation_loaders.get('concat', [])
        
        for (rgb_v, y), (ir_v, _), (concat_v, _) in zip(rgb_loader, ir_loader, concat_loader):
            pc_r, sn_r = self.boosters['rgb'].predict_chain(rgb_v)
            pc_i, sn_i = self.boosters['ir'].predict_chain(ir_v)
            pc_c, sn_c = self.boosters['concat'].predict_chain(concat_v)
            
            # Form standard (N, 6) batch arrays matching stacker rules
            batch_vectors = np.column_stack((pc_r, sn_r, pc_i, sn_i, pc_c, sn_c))
            oof_vectors.append(batch_vectors)
            labels.append(y.numpy())
            
        final_vectors = np.concatenate(oof_vectors, axis=0) if oof_vectors else np.array([])
        final_labels = np.concatenate(labels, axis=0) if labels else np.array([])
        
        filepath = os.path.join(save_dir, 'oof_6D_vectors.pkl')
        if final_vectors.size > 0:
            with open(filepath, 'wb') as f:
                 pickle.dump({'X': final_vectors, 'y': final_labels}, f)
            print(f"VRAM cleared. OOF tensors ({final_vectors.shape}) serialized to {filepath}.")
            
        return final_vectors, final_labels
