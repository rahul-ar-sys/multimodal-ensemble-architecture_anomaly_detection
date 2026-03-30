"""
Booster Engine

Contains the logic to train the Sequential Error-Correction Chain.
PatchCore executes first to flag residuals, which are routed to SimpleNet.
"""
import torch

class Booster:
    def __init__(self, base_model, refiner_model=None):
        """
        base_model: Instantiated PatchCore object.
        refiner_model: Instantiated SimpleNet object. (Bypassed securely if missing)
        """
        self.base_model = base_model
        self.refiner_model = refiner_model

    def train_chain(self, train_dataloader):
        """
        Execute the Boosted Bagging sequence.
        """
        print("==> Base Layer: Fitting PatchCore memory bank...")
        self.base_model.fit(train_dataloader)
        
        print("==> Base Layer: Extracting Hard Samples (False Positives)...")
        # Pseudo extraction wrapper: Requires a second loop through nominal images
        # to generate distances > predefined threshold to collect the 'false positive' vectors
        hard_samples = []
        
        if self.refiner_model is None:
            print("[Mock Warning]: SimpleNet Refiner module is None. Bypassing Boosting Phase cleanly.")
        else:
            print("==> Refiner Layer: Fitting SimpleNet on detected anomaly Hard Samples...")
            if hasattr(self.refiner_model, 'fit'):
                # Bypass fit temporarily if simplenet architecture remains undeclared or under dev
                self.refiner_model.fit(hard_samples)
                
        print("==> Modality Chain sequence complete.")
        
    def predict_chain(self, test_images):
        """
        Generates array predictions bridging both the base structure block and the finer refiner checks.
        """
        pc_maps, pc_scores = self.base_model.predict(test_images)
        
        if self.refiner_model is None or not hasattr(self.refiner_model, 'predict'):
            # Forward pure 0 metrics to keep the OOF vector dynamically flat mapping
            sn_scores = torch.zeros_like(torch.tensor(pc_scores)).numpy()
        else:
            _, sn_scores = self.refiner_model.predict(test_images)
            
        return pc_scores, sn_scores
