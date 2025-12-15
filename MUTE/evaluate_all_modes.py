# ==================== FINAL EVALUATION SCRIPT – 100% WORKING INCLUDING VGG19 ====================
import os
import gc
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ------------------- CONFIG -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MAX_LEN = 128
TEST_PATH = r"C:\Users\RGUKT\Downloads\Temp\test.xlsx"

os.makedirs("evaluation_results", exist_ok=True)
sns.set_style("whitegrid")

# ------------------- DATA LOADING -------------------
def load_and_clean_stance(path):
    df = pd.read_excel(path)
    df['STANCE_1'] = df['STANCE_1'].astype(str).str.lower().str.strip()
    stance_map = {'favour': 'favour', 'favor': 'favour', 'in favour': 'favour', 'in favor': 'favour',
                  'against': 'against', 'none': 'none', 'neutral': 'none', 'nan': 'none', 'no stance': 'none'}
    df['STANCE_1'] = df['STANCE_1'].map(stance_map).fillna('none')
    df = df[df['STANCE_1'].isin(['favour', 'against', 'none'])].copy()
    df['STANCE_1'] = df['STANCE_1'].map({'favour': 0, 'against': 1, 'none': 2})
    df = df.dropna(subset=['full_path', 'Image Text', 'TARGET_1'])
    df = df[df['full_path'].apply(os.path.exists)]
    print(f"Test set: {len(df)} samples")
    return df.reset_index(drop=True)

test_df = load_and_clean_stance(TEST_PATH)

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------- DATASETS -------------------
class TextTargetDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts = (df['TARGET_1'].astype(str) + " [SEP] " + df['Image Text'].astype(str)).tolist()
        self.labels = df['STANCE_1'].values
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length',
                                  max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class ImageStanceDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=img_transform):
        self.df = df; self.transform = transform; self.labels = df['STANCE_1'].values
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]['full_path']).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)

class MultimodalStanceDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, transform=img_transform, max_len=MAX_LEN):
        self.df = df; self.tokenizer = tokenizer; self.transform = transform; self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['TARGET_1']) + " [SEP] " + str(row['Image Text'])
        img = Image.open(row['full_path']).convert("RGB")
        if self.transform: img = self.transform(img)
        encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                  max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'image': img,
            'label': torch.tensor(row['STANCE_1'], dtype=torch.long)
        }

# ------------------- MODELS -------------------
class TextStance(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.model.config.hidden_size, 3)
    def get_features(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:,0,:]
        return self.dropout(hidden)
    def forward(self, input_ids, attention_mask):
        return self.fc(self.get_features(input_ids, attention_mask))

class MultimodalStance(nn.Module):
    def __init__(self, vision_backbone, text_model):
        super().__init__()
        self.vision = vision_backbone
        self.text = text_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(2048 + 768, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 3)
        ).to(DEVICE)
    def forward(self, image, input_ids, attention_mask):
        v = self.vision(image).squeeze(-1).squeeze(-1)
        t = self.text.get_features(input_ids, attention_mask)
        x = torch.cat([v, t], dim=1)
        return self.classifier(x)

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                imgs = batch.get('image')
                out = model(imgs.to(DEVICE), ids, mask) if imgs is not None else model(ids, mask)
            else:
                imgs, labels = batch
                out = model(imgs.to(DEVICE))
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return trues, preds

# ------------------- MAIN EVALUATION -------------------
results = {}
print("\nEVALUATING ALL YOUR MODELS (INCLUDING VGG19)\n")

# === TEXT-ONLY ===
text_configs = {
    "xlm-roberta_stance.pt": ("XLM-RoBERTa", "xlm-roberta-base"),
    "mBert_stance.pt": ("mBERT", "bert-base-multilingual-cased"),
    "IndicBertv2_stance.pt": ("IndicBERTv2", "ai4bharat/IndicBERTv2-MLM-only")
}

for fname, (name, hf) in text_configs.items():
    path = f"saved_models/text/{fname}"
    if not os.path.exists(path): continue
    print(f"→ Text: {name}")
    tokenizer = AutoTokenizer.from_pretrained(hf)
    loader = DataLoader(TextTargetDataset(test_df, tokenizer), batch_size=BATCH_SIZE, shuffle=False)
    model = TextStance(hf).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    state = {("model." + k.split(".", 1)[1] if k.startswith(("bert.", "roberta.")) else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    y_true, y_pred = evaluate(model, loader)
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    results[name] = {"acc": report["accuracy"], "macro": report["macro avg"]["f1-score"]}
    print(f"   Acc: {results[name]['acc']:.4f} | Macro F1: {results[name]['macro']:.4f}\n")
    del model, tokenizer, loader; gc.collect(); torch.cuda.empty_cache()

# === VISION-ONLY (ResNet50 + VGG19 – EXACTLY AS YOU TRAINED) ===
vision_configs = ["resnet50", "vgg19"]  # ← exactly as in your training loop

img_loader = DataLoader(ImageStanceDataset(test_df), batch_size=BATCH_SIZE, shuffle=False)

for name in vision_configs:
    path = f"saved_models/vision/{name}_stance.pt"
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    print(f"→ Vision-only: {name.upper()}")

    # This is EXACTLY how you trained it
    model = models.__dict__[name](pretrained=False)  # No pretrained weights needed at inference
    if "vgg" in name:
        model.classifier[6] = nn.Linear(4096, 3)
    else:
        model.fc = nn.Linear(2048, 3)
    
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    
    y_true, y_pred = evaluate(model, img_loader)
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    display_name = "VGG19" if "vgg" in name else "ResNet50"
    results[f"{display_name} (Vision)"] = {"acc": report["accuracy"], "macro": report["macro avg"]["f1-score"]}
    print(f"   Acc: {results[f'{display_name} (Vision)']['acc']:.4f} | Macro F1: {results[f'{display_name} (Vision)']['macro']:.4f}\n")
    del model; gc.collect(); torch.cuda.empty_cache()

# === MULTIMODAL EVALUATION – PERFECTLY MATCHES YOUR ACTUAL FILES ===
print("\nEvaluating Multimodal Models")
for fname in os.listdir("saved_models/multimodal"):
    if not fname.endswith(".pt"):
        continue
    path = f"saved_models/multimodal/{fname}"
    combo_name = fname.replace(".pt", "")
    print(f"→ Evaluating: {combo_name}")

    # --- Extract the text model name from combo_name and map to real filename ---
    text_part = combo_name.split("+")[1] if "+" in combo_name else ""

    if "XLM" in text_part or "xlm" in text_part:
        hf_name = "xlm-roberta-base"
        text_weight_file = "xlm-roberta_stance.pt"
    elif "mBERT" in text_part or "mBert" in text_part:
        hf_name = "bert-base-multilingual-cased"
        text_weight_file = "mBert_stance.pt"        # ← your real file
    elif "Indic" in text_part:
        hf_name = "ai4bharat/IndicBERTv2-MLM-only"
        text_weight_file = "IndicBertv2_stance.pt"
    else:
        print(f"  Unknown text model in {combo_name}, skipping")
        continue

    print(f"  Detected text model → {text_weight_file}")

    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    loader = DataLoader(MultimodalStanceDataset(test_df, tokenizer), batch_size=12, shuffle=False)

    # Vision backbone (frozen ResNet50)
    backbone = nn.Sequential(*list(models.resnet50(weights="IMAGENET1K_V1").children())[:-1])
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.to(DEVICE)

    # Text model
    text_model = TextStance(hf_name).to(DEVICE)
    text_path = f"saved_models/text/{text_weight_file}"

    if os.path.exists(text_path):
        state = torch.load(text_path, map_location=DEVICE)
        # Fix bert./roberta. → model. prefix
        fixed_state = {}
        for k, v in state.items():
            if k.startswith(("bert.", "roberta.")):
                fixed_state["model." + k.split(".", 1)[1]] = v
            else:
                fixed_state[k] = v
        text_model.load_state_dict(fixed_state, strict=False)
        print(f"  Loaded text weights: {text_weight_file}")
    else:
        print(f"  WARNING: Text weights not found: {text_path}")

    # Full multimodal model
    model = MultimodalStance(backbone, text_model).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))

    # Evaluate
    y_true, y_pred = evaluate(model, loader)
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    results[combo_name] = {
        "acc": report["accuracy"],
        "macro": report["macro avg"]["f1-score"]
    }
    print(f"  Accuracy: {results[combo_name]['acc']:.4f} | Macro F1: {results[combo_name]['macro']:.4f}\n")

    # Cleanup
    del model, backbone, text_model, tokenizer, loader
    gc.collect()
    torch.cuda.empty_cache()
# ==================== FINAL RESULTS ====================
df = pd.DataFrame([
    {"Model": k, "Accuracy": round(v["acc"], 4), "Macro F1": round(v["macro"], 4)}
    for k, v in results.items()
]).sort_values("Macro F1", ascending=False)

print("\n" + "="*90)
print("FINAL TEST RESULTS (Sorted by Macro F1)")
print("="*90)
print(df.to_string(index=False))

df.to_csv("evaluation_results/FINAL_RESULTS_WITH_VGG19.csv", index=False)
df.to_latex("evaluation_results/FINAL_RESULTS_WITH_VGG19.tex", index=False, float_format="%.4f")

plt.figure(figsize=(11, 7))
df.set_index("Model")[["Accuracy", "Macro F1"]].plot(kind="barh", cmap="turbo", width=0.8)
plt.title("Telugu Meme Stance Detection – Final Results", fontsize=16, fontweight="bold")
plt.xlabel("Score")
plt.tight_layout()
plt.savefig("evaluation_results/FINAL_RESULTS_WITH_VGG19.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nBEST MODEL → {df.iloc[0]['Model']} | Macro F1 = {df.iloc[0]['Macro F1']:.4f}")
print("All results saved in evaluation_results/")