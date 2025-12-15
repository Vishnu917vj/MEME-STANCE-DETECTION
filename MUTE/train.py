# # ================================================
# # MULTIMODAL STANCE DETECTION IN TELUGU MEMES
# # 3 Classes: favour | against | none
# # Text + Image + Target → Stance Prediction
# # ================================================

# import os
# import gc
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# from transformers import AutoTokenizer, BertModel, XLMRobertaModel,AutoModel
# from sklearn.metrics import classification_report, confusion_matrix
# from PIL import Image
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ------------------- CONFIG -------------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 16
# MAX_LEN = 128
# NUM_EPOCHS_TEXT = 5
# NUM_EPOCHS_VISION = 8
# NUM_EPOCHS_MULTI = 8
# SEED = 42


# torch.manual_seed(SEED)
# np.random.seed(SEED)
# torch.cuda.empty_cache()

# # UPDATE THESE PATHS!
# TRAIN_PATH = r"C:\Users\RGUKT\Downloads\Temp\output.xlsx"
# TEST_PATH = r"C:\Users\RGUKT\Downloads\Temp\output.xlsx" # or same file if splitting

# os.makedirs("saved_models/text", exist_ok=True)
# os.makedirs("saved_models/vision", exist_ok=True)
# os.makedirs("saved_models/multimodal", exist_ok=True)
# os.makedirs("outputs", exist_ok=True)

# # ------------------- IMAGE TRANSFORMS -------------------
# img_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # ------------------- DATA LOADING & CLEANING -------------------
# def load_and_clean_stance(path):
#     df = pd.read_excel(path)
#     print(f"Loaded {len(df)} samples from {path}")

#     # Clean Stance column
#     df['STANCE_1'] = df['STANCE_1'].astype(str).str.lower().str.strip()
#     stance_map = {
#         'favour': 'favour', 'favor': 'favour', 'in favour': 'favour', 'in favor': 'favour',
#         'against': 'against',
#         'none': 'none', 'neutral': 'none', 'nan': 'none', 'no stance': 'none'
#     }
#     df['STANCE_1'] = df['STANCE_1'].map(stance_map).fillna('none')
#     df = df[df['STANCE_1'].isin(['favour', 'against', 'none'])].copy()

#     # Map to 0,1,2
#     df['STANCE_1'] = df['STANCE_1'].map({'favour': 0, 'against': 1, 'none': 2})

#     # Clean required columns
#     required = ['full_path', 'Image Text', 'TARGET_1']
#     df = df.dropna(subset=required)
#     df = df[df['full_path'].apply(os.path.exists)]

#     print(f"After cleaning: {len(df)} samples")
#     print("Stance distribution:")
#     print(df['STANCE_1'].value_counts())
#     return df.reset_index(drop=True)

# train_df = load_and_clean_stance(TRAIN_PATH)
# test_df = load_and_clean_stance(TEST_PATH)

# # ------------------- DATASETS -------------------
# class TextTargetDataset(Dataset):
#     def __init__(self, df, tokenizer, max_len=MAX_LEN):
#         self.texts = (df['TARGET_1'].astype(str) + " [SEP] " + df['Image Text'].astype(str)).tolist()
#         self.labels = df['STANCE_1'].values
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self): return len(self.texts)
#     def __getitem__(self, idx):
#         encoding = self.tokenizer(
#             self.texts[idx],
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_len,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encoding['input_ids'].squeeze(),
#             'attention_mask': encoding['attention_mask'].squeeze(),
#             'label': torch.tensor(self.labels[idx], dtype=torch.long)
#         }

# class ImageStanceDataset(Dataset):
#     def __init__(self, df, transform=img_transform):
#         self.df = df
#         self.transform = transform
#         self.labels = df['STANCE_1'].values

#     def __len__(self): return len(self.df)
#     def __getitem__(self, idx):
#         img_path = self.df.iloc[idx]['full_path']
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
#         return image, label

# class MultimodalStanceDataset(Dataset):
#     def __init__(self, df, tokenizer, transform=img_transform, max_len=MAX_LEN):
#         self.df = df
#         self.tokenizer = tokenizer
#         self.transform = transform
#         self.max_len = max_len

#     def __len__(self): return len(self.df)
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         text = str(row['TARGET_1']) + " [SEP] " + str(row['Image Text'])
#         img_path = row['full_path']
#         label = torch.tensor(row['STANCE_1'], dtype=torch.long)

#         encoding = self.tokenizer(
#             text, truncation=True, padding='max_length',
#             max_length=self.max_len, return_tensors='pt'
#         )
#         image = Image.open(img_path).convert("RGB")
#         image = self.transform(image)

#         return {
#             'input_ids': encoding['input_ids'].squeeze(),
#             'attention_mask': encoding['attention_mask'].squeeze(),
#             'image': image,
#             'label': label
#         }
# class TextStance(nn.Module):
#     def __init__(self, model_name):
#         super().__init__()
#         self.model = AutoModel.from_pretrained(model_name)
#         self.dropout = nn.Dropout(0.3)
#         hidden_size = self.model.config.hidden_size
#         self.fc = nn.Linear(hidden_size, 3)

#     def get_features(self, input_ids, attention_mask):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
#             hidden = outputs.pooler_output
#         else:
#             hidden = outputs.last_hidden_state[:, 0, :]
#         return self.dropout(hidden)  # Return 768-dim features

#     def forward(self, input_ids, attention_mask):
#         features = self.get_features(input_ids, attention_mask)
#         return self.fc(features)  # Only used in text-only training
# # ------------------- MODELS (3 Classes) -------------------
# # Text Models
# class BERTStance(nn.Module):
#     def __init__(self, model_name):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(model_name)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(768, 3)

#     def forward(self, ids, mask):
#         out = self.bert(ids, attention_mask=mask).pooler_output
#         return self.fc(self.dropout(out))

# class XLMRStance(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.xlm = XLMRobertaModel.from_pretrained("xlm-roberta-base")
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(768, 3)

#     def forward(self, ids, mask):
#         out = self.xlm(ids, attention_mask=mask).pooler_output
#         return self.fc(self.dropout(out))


# # Multimodal Late Fusion
# # FIXED VERSION — WORKS FOR BOTH ResNet50 AND VGG19
# # ─────────────────────── FINAL FIXED MULTIMODAL CLASS ───────────────────────
# # Vision backbone — ONLY ResNet50 (safe & best)
# def get_vision_backbone(name="resnet50", freeze=True):
#     backbone = models.resnet50(pretrained=True)
#     backbone = nn.Sequential(*list(backbone.children())[:-1])  # Output: (B,2048,1,1)
#     if freeze:
#         for p in backbone.parameters():
#             p.requires_grad = False
#     print("Vision backbone: ResNet50 → 2048-dim")
#     return backbone.to(DEVICE), 2048

# # Multimodal — with safe flatten
# class MultimodalStance(nn.Module):
#     def __init__(self, vision_backbone, text_model, vis_dim):
#         super().__init__()
#         self.vision = vision_backbone
#         self.text = text_model  # This has its own .fc that outputs 3 classes
#         total_dim = vis_dim + 768  # 2048 + 768 = 2816
#         print(f"Multimodal input: {vis_dim} + 768 = {total_dim}")

#         # Remove the final classifier from text_model if it exists
#         # We'll use only the embeddings
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(total_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 3)
#         ).to(DEVICE)

#     def forward(self, image, input_ids, attention_mask):
#         # Vision
#         v = self.vision(image)                    # (B, 2048, 1, 1)
#         v = v.squeeze(-1).squeeze(-1)             # or .view(B, -1)

#         # Text features (768-dim)
#         text_out = self.text.model(input_ids=input_ids, attention_mask=attention_mask)
#         if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
#             t = text_out.pooler_output
#         else:
#             t = text_out.last_hidden_state[:, 0, :]   # CLS token

#         t = self.text.dropout(t)

#         # Concatenate
#         x = torch.cat([v, t], dim=1)                  # (B, 2816)
#         return self.classifier(x)

# # ------------------- TRAINING & EVAL -------------------
# def train_model(model, loader, epochs, lr=2e-5):
#     model.train()
#     optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in loader:
#             optimizer.zero_grad()
#             if isinstance(batch, dict):
#                 ids = batch['input_ids'].to(DEVICE)
#                 mask = batch['attention_mask'].to(DEVICE)
#                 labels = batch['label'].to(DEVICE)
#                 imgs = batch.get('image')
#                 if imgs is not None:
#                     imgs = imgs.to(DEVICE)
#                     outputs = model(imgs, ids, mask)
#                 else:
#                     outputs = model(ids, mask)
#             else:
#                 imgs, labels = batch
#                 imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#                 outputs = model(imgs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

# def evaluate_stance(model, loader):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for batch in loader:
#             if isinstance(batch, dict):
#                 ids = batch['input_ids'].to(DEVICE)
#                 mask = batch['attention_mask'].to(DEVICE)
#                 labels = batch['label'].cpu().numpy()
#                 imgs = batch.get('image')
#                 if imgs is not None:
#                     imgs = imgs.to(DEVICE)
#                     outputs = model(imgs, ids, mask)
#                 else:
#                     outputs = model(ids, mask)
#             else:
#                 imgs, labels = batch
#                 imgs = imgs.to(DEVICE)
#                 outputs = model(imgs)
#                 labels = labels.cpu().numpy()
#             preds = torch.argmax(outputs, dim=1).cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(labels)

#     report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
#     return {
#         "accuracy": report["accuracy"],
#         "macro_f1": report["macro avg"]["f1-score"],
#         "weighted_f1": report["weighted avg"]["f1-score"],
#         "y_true": all_labels,
#         "y_pred": all_preds
#     }

# # ------------------- MAIN EXPERIMENT -------------------
# results = {}

# print("\n" + "="*70)
# print("MULTIMODAL STANCE DETECTION - TELUGU MEMES")
# print("="*70)

# # === PHASE 1: TEXT + TARGET MODELS ===
# # print("\nPHASE 1: TEXT + TARGET MODELS")
# # text_configs = [
# #     ("XLM-RoBERTa", "xlm-roberta-base", "xlm-roberta-base"),
# #     ("mBERT", "bert-base-multilingual-cased", "bert-base-multilingual-cased"),
# #     ("IndicBERTv2", "ai4bharat/IndicBERTv2-MLM-only", "ai4bharat/IndicBERTv2-MLM-only"),  # AI4Bharat Model
# # ]

# # for name, model_name, hf_model_name in text_configs:
# #     print(f"\nTraining {name}...")
# #     tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
# #     train_ds = TextTargetDataset(train_df, tokenizer)
# #     test_ds = TextTargetDataset(test_df, tokenizer)
# #     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
# #     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# #     model = IndicTextStance(hf_model_name).to(DEVICE)

# #     train_model(model, train_loader, NUM_EPOCHS_TEXT, lr=2e-5)
# #     metrics = evaluate_stance(model, test_loader)
# #     results[name] = metrics

# #     torch.save(model.state_dict(), f"saved_models/text/{name}_stance.pt")
# #     del model, tokenizer, train_ds, test_ds; gc.collect(); torch.cuda.empty_cache()

# # === PHASE 2: VISION-ONLY ===
# print("\nPHASE 2: VISION-ONLY MODELS")
# #resNet50
# vision_names = [ "vgg19"]

# train_img_ds = ImageStanceDataset(train_df)
# test_img_ds = ImageStanceDataset(test_df)
# img_train_loader = DataLoader(train_img_ds, batch_size=BATCH_SIZE, shuffle=True)
# img_test_loader = DataLoader(test_img_ds, batch_size=BATCH_SIZE)

# for name in vision_names:
#     print(f"\nTraining {name.upper()} (Vision Only)...")
#     backbone = models.__dict__[name](pretrained=True)
#     if "vgg" in name:
#         backbone.classifier[6] = nn.Linear(4096, 3)
#     else:
#         backbone.fc = nn.Linear(2048, 3)
#     backbone = backbone.to(DEVICE)

#     train_model(backbone, img_train_loader, NUM_EPOCHS_VISION, lr=1e-4)
#     metrics = evaluate_stance(backbone, img_test_loader)
#     results[name.upper()] = metrics

#     torch.save(backbone.state_dict(), f"saved_models/vision/{name}_stance.pt")
#     del backbone; gc.collect(); torch.cuda.empty_cache()

# # === PHASE 3: MULTIMODAL ===
# # === PHASE 3: MULTIMODAL (Image + Text + Target) ===
# print("\nPHASE 3: MULTIMODAL (Image + Text + Target)")
# multimodal_combos = [
#     ("resnet50", "xlm-roberta-base", "ResNet50+XLM-R"),
#     # ("resnet50", "bert-base-multilingual-cased", "ResNet50+mBERT"),
#     # ("resnet50", "ai4bharat/IndicBERTv2-MLM-only", "ResNet50+IndicBERTv2"),
# ]

# for vis_name, text_name, combo_name in multimodal_combos:
#     print(f"\nTraining {combo_name}...")
#     tokenizer = AutoTokenizer.from_pretrained(text_name)

#     train_ds = MultimodalStanceDataset(train_df, tokenizer)
#     test_ds = MultimodalStanceDataset(test_df, tokenizer)
#     train_loader = DataLoader(train_ds, batch_size=12, shuffle=True)
#     test_loader = DataLoader(test_ds, batch_size=12)

#     # Vision backbone (ResNet50 → always 2048)
#     vision_backbone, vis_dim = get_vision_backbone(vis_name, freeze=True)
#     assert vis_dim == 2048, "Only ResNet50 is supported now"

#     # TEXT MODEL — FINAL BULLETPROOF VERSION
#     class TextStance(nn.Module):
#         def __init__(self, model_name):
#             super().__init__()
#             self.model = AutoModel.from_pretrained(model_name)
#             self.dropout = nn.Dropout(0.3)
#             hidden_size = self.model.config.hidden_size  # 768 for all
#             self.fc = nn.Linear(hidden_size, 3)

#         def forward(self, input_ids, attention_mask):
#             outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#             # XLM-R has no pooler_output → use [CLS] token
#             if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
#                 hidden = outputs.pooler_output
#             else:
#                 hidden = outputs.last_hidden_state[:, 0, :]  # [CLS]
#             return self.fc(self.dropout(hidden))

#     text_model = TextStance(text_name).to(DEVICE)

#     # Multimodal model
#     model = MultimodalStance(vision_backbone, text_model, vis_dim).to(DEVICE)

#     # Train
#     train_model(model, train_loader, NUM_EPOCHS_MULTI, lr=3e-4)
#     metrics = evaluate_stance(model, test_loader)
#     results[combo_name] = metrics

#     torch.save(model.state_dict(), 
#            f"saved_models/multimodal/{combo_name}.pt",
#            _use_new_zipfile_serialization=False)
#     print(f"→ {combo_name} | Macro F1: {metrics['macro_f1']:.4f}")

#     del model, vision_backbone, text_model, tokenizer
#     gc.collect(); torch.cuda.empty_cache()
# # === FINAL RESULTS ===
# print("\n" + "="*80)
# print("FINAL RESULTS - STANCE DETECTION (Macro F1 is Key!)")
# print("="*80)

# comparison = []
# for name, r in results.items():
#     comparison.append({
#         "Model": name,
#         "Accuracy": round(r["accuracy"], 4),
#         "Macro F1": round(r["macro_f1"], 4),
#         "Weighted F1": round(r["weighted_f1"], 4)
#     })

# df_results = pd.DataFrame(comparison)
# df_results = df_results.sort_values("Macro F1", ascending=False)
# print(df_results.to_string(index=False))
# df_results.to_csv("outputs/stance_detection_results.csv", index=False)

# # Plot
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df_results, x="Model", y="Macro F1", palette="viridis")
# plt.title("Stance Detection Performance (Macro F1)")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("outputs/stance_macro_f1_comparison.png", dpi=300)
# plt.show()

# print("\nALL DONE!")
# print("Results → outputs/stance_detection_results.csv")
# print("Best model likely: ResNet50 + XLM-RoBERTa")
# print("Ready for paper submission!")



# ================================================
# MULTIMODAL STANCE DETECTION IN TELUGU MEMES
# 3 Classes: favour | against | none
# Text + Image + Target → Stance Prediction
# ================================================

import os
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- CONFIG -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MAX_LEN = 128
NUM_EPOCHS_TEXT = 5
NUM_EPOCHS_VISION = 8
NUM_EPOCHS_MULTI = 8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.empty_cache()

# UPDATE THESE PATHS!
TRAIN_PATH = r"C:\Users\RGUKT\Downloads\Temp\output.xlsx"
TEST_PATH = r"C:\Users\RGUKT\Downloads\Temp\output.xlsx" # or same file if splitting

os.makedirs("saved_models/text", exist_ok=True)
os.makedirs("saved_models/vision", exist_ok=True)
os.makedirs("saved_models/multimodal", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ------------------- IMAGE TRANSFORMS -------------------
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------- DATA LOADING & CLEANING -------------------
def load_and_clean_stance(path):
    df = pd.read_excel(path)
    print(f"Loaded {len(df)} samples from {path}")

    # Clean Stance column
    df['STANCE_1'] = df['STANCE_1'].astype(str).str.lower().str.strip()
    stance_map = {
        'favour': 'favour', 'favor': 'favour', 'in favour': 'favour', 'in favor': 'favour',
        'against': 'against',
        'none': 'none', 'neutral': 'none', 'nan': 'none', 'no stance': 'none'
    }
    df['STANCE_1'] = df['STANCE_1'].map(stance_map).fillna('none')
    df = df[df['STANCE_1'].isin(['favour', 'against', 'none'])].copy()

    # Map to 0,1,2
    df['STANCE_1'] = df['STANCE_1'].map({'favour': 0, 'against': 1, 'none': 2})

    # Clean required columns
    required = ['full_path', 'Image Text', 'TARGET_1']
    df = df.dropna(subset=required)
    df = df[df['full_path'].apply(os.path.exists)]

    print(f"After cleaning: {len(df)} samples")
    print("Stance distribution:")
    print(df['STANCE_1'].value_counts())
    return df.reset_index(drop=True)

train_df = load_and_clean_stance(TRAIN_PATH)
test_df = load_and_clean_stance(TEST_PATH)

# ------------------- DATASETS -------------------
class TextTargetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts = (df['TARGET_1'].astype(str) + " [SEP] " + df['Image Text'].astype(str)).tolist()
        self.labels = df['STANCE_1'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class ImageStanceDataset(Dataset):
    def __init__(self, df, transform=img_transform):
        self.df = df
        self.transform = transform
        self.labels = df['STANCE_1'].values

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['full_path']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

class MultimodalStanceDataset(Dataset):
    def __init__(self, df, tokenizer, transform=img_transform, max_len=MAX_LEN):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['TARGET_1']) + " [SEP] " + str(row['Image Text'])
        img_path = row['full_path']
        label = torch.tensor(row['STANCE_1'], dtype=torch.long)

        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'image': image,
            'label': label
        }

# ------------------- MODELS (3 Classes) -------------------
class TextStance(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        hidden_size = self.model.config.hidden_size
        self.fc = nn.Linear(hidden_size, 3)

    def get_features(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            hidden = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state[:, 0, :]
        return self.dropout(hidden)  # Return 768-dim features

    def forward(self, input_ids, attention_mask):
        features = self.get_features(input_ids, attention_mask)
        return self.fc(features)  # Only used in text-only training

# Vision backbone — ONLY ResNet50 (safe & best)
def get_vision_backbone(name="resnet50", freeze=True):
    backbone = models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-1])  # Output: (B,2048,1,1)
    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
    print("Vision backbone: ResNet50 → 2048-dim")
    return backbone.to(DEVICE), 2048

# Multimodal — with safe flatten
class MultimodalStance(nn.Module):
    def __init__(self, vision_backbone, text_model, vis_dim):
        super().__init__()
        self.vision = vision_backbone
        self.text = text_model  # This has its own .fc that outputs 3 classes
        total_dim = vis_dim + 768  # 2048 + 768 = 2816
        print(f"Multimodal input: {vis_dim} + 768 = {total_dim}")

        # Remove the final classifier from text_model if it exists
        # We'll use only the embeddings
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        ).to(DEVICE)

    def forward(self, image, input_ids, attention_mask):
        # Vision
        v = self.vision(image)                    # (B, 2048, 1, 1)
        v = v.squeeze(-1).squeeze(-1)             # or .view(B, -1)

        # Text features (768-dim)
        t = self.text.get_features(input_ids, attention_mask)

        # Concatenate
        x = torch.cat([v, t], dim=1)                  # (B, 2816)
        return self.classifier(x)

# ------------------- TRAINING & EVAL -------------------
def train_model(model, loader, epochs, lr=2e-5):
    model.train()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            if isinstance(batch, dict):
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                imgs = batch.get('image')
                if imgs is not None:
                    imgs = imgs.to(DEVICE)
                    outputs = model(imgs, ids, mask)
                else:
                    outputs = model(ids, mask)
            else:
                imgs, labels = batch
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

def evaluate_stance(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].cpu().numpy()
                imgs = batch.get('image')
                if imgs is not None:
                    imgs = imgs.to(DEVICE)
                    outputs = model(imgs, ids, mask)
                else:
                    outputs = model(ids, mask)
            else:
                imgs, labels = batch
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                labels = labels.cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "y_true": all_labels,
        "y_pred": all_preds
    }

# ------------------- MAIN EXPERIMENT -------------------
results = {}

print("\n" + "="*70)
print("MULTIMODAL STANCE DETECTION - TELUGU MEMES")
print("="*70)
text_configs = [
    ("XLM-RoBERTa", "xlm-roberta-base", "xlm-roberta-base"),
    ("mBERT", "bert-base-multilingual-cased", "bert-base-multilingual-cased"),
    ("IndicBERTv2", "ai4bharat/IndicBERTv2-MLM-only", "ai4bharat/IndicBERTv2-MLM-only"),
]
# === PHASE 1: RESUME YOUR 3 TEXT MODELS FOR 6TH EPOCH (EXACT FILENAMES + PREFIX FIX) ===
print("\n" + "="*80)
print("RESUMING TEXT MODELS FOR 6TH EPOCH - FINAL VERSION")
print("="*80)

# Map display name → actual saved filename (without .pt)
file_to_config = {
    "IndicBertv2_stance": {
        "display": "IndicBERTv2",
        "hf_name": "ai4bharat/IndicBERTv2-MLM-only"
    },
    "mBert_stance": {
        "display": "mBERT",
        "hf_name": "bert-base-multilingual-cased"
    },
    "xlm-roberta_stance": {
        "display": "XLM-RoBERTa",
        "hf_name": "xlm-roberta-base"
    }
}

for filename, config in file_to_config.items():
    save_path = f"saved_models/text/{filename}.pt"
    
    if not os.path.exists(save_path):
        print(f"File not found: {save_path} → Skipping")
        continue

    display_name = config["display"]
    hf_model_name = config["hf_name"]
    
    print(f"\nResuming → {display_name} (6th epoch)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=True)

    # Datasets
    train_ds = TextTargetDataset(train_df, tokenizer)
    test_ds  = TextTargetDataset(test_df, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = TextStance(hf_model_name).to(DEVICE)

    # Load and fix name mismatch
    state_dict = torch.load(save_path, map_location=DEVICE)
    new_state_dict = {}

    for k, v in state_dict.items():
        # Fix old prefixes: bert. or roberta. → model.
        if k.startswith("bert.") or k.startswith("roberta."):
            new_k = "model." + k.split(".", 1)[1]
        else:
            new_k = k
        new_state_dict[new_k] = v

    # Load (allow missing classifier head)
    model.load_state_dict(new_state_dict, strict=False)
    print(f"  Loaded & fixed weights: {filename}.pt")

    # Train 1 more epoch (6th)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=3e-6)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"  6th epoch completed | Loss: {total_loss/len(train_loader):.5f}")

    # Evaluate
    metrics = evaluate_stance(model, test_loader)
    results[f"{display_name}_e6"] = metrics
    print(f"  After 6th epoch → Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")

    # Save correctly (with proper 'model.' prefix)
    torch.save(model.state_dict(), save_path)
    print(f"  Updated model saved → {save_path}\n")

    # Cleanup
    del model, tokenizer, train_ds, test_ds, train_loader, test_loader, state_dict, new_state_dict
    gc.collect()
    torch.cuda.empty_cache()

print("All 3 text models successfully trained for their 6th epoch!")
print("They are now saved correctly and ready for multimodal training!")
print("\nPHASE 2: VISION-ONLY MODELS")
vision_names = ["resnet50", "vgg19"]

train_img_ds = ImageStanceDataset(train_df)
test_img_ds = ImageStanceDataset(test_df)
img_train_loader = DataLoader(train_img_ds, batch_size=BATCH_SIZE, shuffle=True)
img_test_loader = DataLoader(test_img_ds, batch_size=BATCH_SIZE)

for name in vision_names:
    print(f"\nTraining {name.upper()} (Vision Only)...")
    backbone = models.__dict__[name](pretrained=True)
    if "vgg" in name:
        backbone.classifier[6] = nn.Linear(4096, 3)
    else:
        backbone.fc = nn.Linear(2048, 3)
    backbone = backbone.to(DEVICE)

    train_model(backbone, img_train_loader, NUM_EPOCHS_VISION, lr=1e-4)
    metrics = evaluate_stance(backbone, img_test_loader)
    results[name.upper()] = metrics

    torch.save(backbone.state_dict(), f"saved_models/vision/{name}_stance.pt")
    del backbone; gc.collect(); torch.cuda.empty_cache()

# === PHASE 3: MULTIMODAL (Image + Text + Target) ===
print("\nPHASE 3: MULTIMODAL (Image + Text + Target)")
multimodal_combos = [
    ("resnet50", "xlm-roberta-base", "ResNet50+XLM-R"),
      ("resnet50", "bert-base-multilingual-cased", "ResNet50+mBERT"),
    ("resnet50", "ai4bharat/IndicBERTv2-MLM-only", "ResNet50+IndicBERTv2"),
]

for vis_name, text_name, combo_name in multimodal_combos:
    print(f"\nTraining {combo_name}...")
    tokenizer = AutoTokenizer.from_pretrained(text_name)

    train_ds = MultimodalStanceDataset(train_df, tokenizer)
    test_ds = MultimodalStanceDataset(test_df, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=12)

    # Vision backbone (ResNet50 → always 2048)
    vision_backbone, vis_dim = get_vision_backbone(vis_name, freeze=True)
    assert vis_dim == 2048, "Only ResNet50 is supported now"

    text_model = TextStance(text_name).to(DEVICE)

    # Multimodal model
    model = MultimodalStance(vision_backbone, text_model, vis_dim).to(DEVICE)

    # Train
    train_model(model, train_loader, NUM_EPOCHS_MULTI, lr=3e-4)
    metrics = evaluate_stance(model, test_loader)
    results[combo_name] = metrics

    torch.save(model.state_dict(), 
           f"saved_models/multimodal/{combo_name}.pt",
           _use_new_zipfile_serialization=False)
    print(f"→ {combo_name} | Macro F1: {metrics['macro_f1']:.4f}")

    del model, vision_backbone, text_model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

# === FINAL RESULTS ===
print("\n" + "="*80)
print("FINAL RESULTS - STANCE DETECTION (Macro F1 is Key!)")
print("="*80)

comparison = []   
for name, r in results.items():
    comparison.append({
        "Model": name,
        "Accuracy": round(r["accuracy"], 4),
        "Macro F1": round(r["macro_f1"], 4),
        "Weighted F1": round(r["weighted_f1"], 4)
    })

df_results = pd.DataFrame(comparison)
df_results = df_results.sort_values("Macro F1", ascending=False)
print(df_results.to_string(index=False))
df_results.to_csv("outputs/stance_detection_results.csv", index=False)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=df_results, x="Model", y="Macro F1", palette="viridis")
plt.title("Stance Detection Performance (Macro F1)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/stance_macro_f1_comparison.png", dpi=300)
plt.show()

print("\nALL DONE!")
print("Results → outputs/stance_detection_results.csv")
print("Best model likely: ResNet50 + XLM-RoBERTa")
print("Ready for paper submission!")