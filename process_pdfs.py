import os
import json
import fitz  # PyMuPDF
import re
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# Use a lightweight model for document understanding
MODEL_NAME = "microsoft/DialoGPT-small"  # ~117MB, we'll use it for text classification
MAX_MODEL_SIZE = 200 * 1024 * 1024  # 200MB in bytes

class DocumentOutlineExtractor:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load lightweight model for text analysis"""
        try:
            # Use a small BERT-like model for text classification
            model_name = "prajjwal1/bert-tiny"  # Only ~17MB
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Falling back to rule-based approach")
            self.tokenizer = None
            self.model = None

    def get_text_embedding(self, text):
        """Get text embedding using the model"""
        if not self.model or not self.tokenizer:
            return None
        
        try:
            # Tokenize and get embeddings
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.numpy().flatten()
        except:
            return None

    def extract_text_with_properties(self, pdf_path):
        """Extract text with detailed properties from PDF"""
        doc = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict", flags=11)  # Get detailed formatting info
            
            page_texts = []
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text and len(text) > 2:  # Filter very short text
                                # Calculate text properties
                                bbox = span["bbox"]
                                page_texts.append({
                                    "text": text,
                                    "font": span["font"],
                                    "size": span["size"],
                                    "flags": span["flags"],
                                    "bbox": bbox,
                                    "page": page_num + 1,
                                    "x": bbox[0],
                                    "y": bbox[1],
                                    "width": bbox[2] - bbox[0],
                                    "height": bbox[3] - bbox[1],
                                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                })
            
            pages_data.append(page_texts)
        
        doc.close()
        return pages_data

    def analyze_document_structure(self, pages_data):
        """Analyze document structure using ML and statistical methods"""
        all_texts = []
        for page in pages_data:
            all_texts.extend(page)
        
        if not all_texts:
            return {}
        
        # Extract features for each text element
        features = []
        texts = []
        
        for text_obj in all_texts:
            # Basic features
            font_size = text_obj["size"]
            is_bold = bool(text_obj["flags"] & 2**4)
            is_italic = bool(text_obj["flags"] & 2**5)
            text_length = len(text_obj["text"])
            word_count = len(text_obj["text"].split())
            
            # Position features (normalized)
            y_pos = text_obj["y"] / 800  # Normalize assuming standard page height
            x_pos = text_obj["x"] / 600  # Normalize assuming standard page width
            
            # Text pattern features
            is_numbered = bool(re.match(r'^\d+\.?\s+', text_obj["text"]))
            is_title_case = text_obj["text"].istitle()
            is_upper_case = text_obj["text"].isupper()
            has_colon = ':' in text_obj["text"]
            
            # Create feature vector
            feature_vector = [
                font_size,
                float(is_bold) * 2,  # Weight bold text more
                float(is_italic),
                text_length,
                word_count,
                y_pos,
                x_pos,
                float(is_numbered) * 2,  # Weight numbered items more
                float(is_title_case),
                float(is_upper_case),
                float(has_colon),
                text_obj["page"]
            ]
            
            features.append(feature_vector)
            texts.append(text_obj)
        
        # Normalize features
        features = np.array(features)
        if features.shape[0] > 0:
            # Simple normalization
            feature_means = np.mean(features, axis=0)
            feature_stds = np.std(features, axis=0)
            feature_stds[feature_stds == 0] = 1  # Avoid division by zero
            features = (features - feature_means) / feature_stds
        
        return {
            "features": features,
            "texts": texts,
            "font_sizes": [t["size"] for t in texts],
            "body_font_size": self.get_body_font_size([t["size"] for t in texts])
        }

    def get_body_font_size(self, font_sizes):
        """Determine the most common font size (likely body text)"""
        if not font_sizes:
            return 12
        
        # Count occurrences of each font size
        size_counts = defaultdict(int)
        for size in font_sizes:
            size_counts[round(size, 1)] += 1
        
        # Return most common size
        return max(size_counts.items(), key=lambda x: x[1])[0]

    def classify_headings(self, analysis_data):
        """Classify text elements as headings using ML approach"""
        if not analysis_data or "features" not in analysis_data:
            return []
        
        features = analysis_data["features"]
        texts = analysis_data["texts"]
        
        if len(features) == 0:
            return []
        
        # Use clustering to identify potential headings
        n_clusters = min(5, len(features))  # Max 5 clusters
        
        if n_clusters < 2:
            # If too few elements, use rule-based approach
            return self.rule_based_heading_detection(texts, analysis_data)
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            
            # Analyze clusters to identify heading clusters
            cluster_analysis = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                cluster_analysis[cluster_id].append({
                    "text_obj": texts[i],
                    "features": features[i]
                })
            
            # Score clusters based on heading-like properties
            heading_candidates = []
            for cluster_id, cluster_items in cluster_analysis.items():
                cluster_score = self.score_cluster_as_heading(cluster_items, analysis_data)
                
                if cluster_score > 0.3:  # Threshold for heading likelihood
                    for item in cluster_items:
                        text_obj = item["text_obj"]
                        if self.validate_heading_text(text_obj["text"]):
                            heading_candidates.append({
                                "text": text_obj["text"],
                                "page": text_obj["page"],
                                "score": cluster_score,
                                "size": text_obj["size"],
                                "font": text_obj["font"],
                                "flags": text_obj["flags"]
                            })
            
            return heading_candidates
            
        except Exception as e:
            print(f"Clustering failed: {e}, falling back to rule-based")
            return self.rule_based_heading_detection(texts, analysis_data)

    def score_cluster_as_heading(self, cluster_items, analysis_data):
        """Score a cluster's likelihood of being headings"""
        if not cluster_items:
            return 0
        
        body_font_size = analysis_data.get("body_font_size", 12)
        
        # Calculate cluster statistics
        font_sizes = [item["text_obj"]["size"] for item in cluster_items]
        avg_font_size = np.mean(font_sizes)
        
        # Text lengths
        text_lengths = [len(item["text_obj"]["text"]) for item in cluster_items]
        avg_text_length = np.mean(text_lengths)
        
        # Bold count
        bold_count = sum(1 for item in cluster_items if item["text_obj"]["flags"] & 2**4)
        bold_ratio = bold_count / len(cluster_items)
        
        # Calculate score
        size_score = min((avg_font_size / body_font_size - 1) * 2, 1) if body_font_size > 0 else 0
        length_score = max(0, 1 - avg_text_length / 100)  # Prefer shorter text
        bold_score = bold_ratio
        
        # Pattern score
        pattern_score = 0
        for item in cluster_items:
            text = item["text_obj"]["text"]
            if re.match(r'^\d+\.?\s+', text):  # Numbered
                pattern_score += 0.2
            if any(keyword in text.lower() for keyword in ['chapter', 'section', 'introduction', 'conclusion']):
                pattern_score += 0.3
        
        pattern_score = min(pattern_score / len(cluster_items), 1)
        
        total_score = (size_score + length_score + bold_score + pattern_score) / 4
        return total_score

    def rule_based_heading_detection(self, texts, analysis_data):
        """Fallback rule-based heading detection"""
        body_font_size = analysis_data.get("body_font_size", 12)
        headings = []
        
        for text_obj in texts:
            if self.is_likely_heading_rule_based(text_obj, body_font_size):
                headings.append({
                    "text": text_obj["text"],
                    "page": text_obj["page"],
                    "score": 0.5,
                    "size": text_obj["size"],
                    "font": text_obj["font"],
                    "flags": text_obj["flags"]
                })
        
        return headings

    def is_likely_heading_rule_based(self, text_obj, body_font_size):
        """Rule-based heading detection"""
        text = text_obj["text"]
        size = text_obj["size"]
        flags = text_obj["flags"]
        
        # Basic filters
        if len(text) < 3 or len(text) > 200:
            return False
        
        if text.isdigit() or re.match(r'^\d+$', text):
            return False
        
        # Size check
        if size < body_font_size * 1.1:
            return False
        
        # Pattern checks
        is_bold = bool(flags & 2**4)
        is_numbered = bool(re.match(r'^\d+\.?\s+', text))
        has_heading_keywords = any(keyword in text.lower() for keyword in 
                                 ['chapter', 'section', 'introduction', 'conclusion', 'summary', 'overview'])
        
        # Scoring
        score = 0
        if size > body_font_size * 1.3:
            score += 1
        if is_bold:
            score += 1
        if is_numbered:
            score += 1
        if has_heading_keywords:
            score += 1
        if text.istitle():
            score += 0.5
        
        return score >= 2

    def validate_heading_text(self, text):
        """Validate if text is appropriate for a heading"""
        if not text or len(text) < 3:
            return False
        
        # Skip obvious non-headings
        skip_patterns = [
            r'^\d+$',  # Just numbers
            r'^page \d+',  # Page numbers
            r'^\d{4}$',  # Years only
            r'www\.',  # URLs
            r'@',  # Email addresses
            r'tel:',  # Phone numbers
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, text.lower()):
                return False
        
        return True

    def assign_heading_levels(self, headings):
        """Assign H1, H2, H3 levels to headings"""
        if not headings:
            return headings
        
        # Sort by score and size for level assignment
        sorted_headings = sorted(headings, key=lambda x: (-x["score"], -x["size"]))
        
        # Assign levels based on ranking
        for i, heading in enumerate(sorted_headings):
            if i < len(sorted_headings) * 0.3:  # Top 30%
                heading["level"] = "H1"
            elif i < len(sorted_headings) * 0.7:  # Next 40%
                heading["level"] = "H2"
            else:  # Bottom 30%
                heading["level"] = "H3"
        
        return headings

    def extract_title(self, pages_data):
        """Extract document title"""
        if not pages_data or not pages_data[0]:
            return "Document"
        
        first_page = pages_data[0]
        
        # Find largest text on first page that looks like a title
        candidates = []
        for text_obj in first_page:
            text = text_obj["text"]
            if 10 <= len(text) <= 150 and self.validate_heading_text(text):
                candidates.append({
                    "text": text,
                    "size": text_obj["size"],
                    "y": text_obj["y"]
                })
        
        if candidates:
            # Sort by size (descending) and position (ascending - higher on page)
            candidates.sort(key=lambda x: (-x["size"], x["y"]))
            return candidates[0]["text"]
        
        return "Document"

    def extract_outline(self, pdf_path):
        """Main function to extract outline"""
        try:
            # Extract text with properties
            pages_data = self.extract_text_with_properties(pdf_path)
            
            if not pages_data:
                return {"title": "Document", "outline": []}
            
            # Analyze document structure
            analysis_data = self.analyze_document_structure(pages_data)
            
            # Extract title
            title = self.extract_title(pages_data)
            
            # Classify headings
            headings = self.classify_headings(analysis_data)
            
            # Assign heading levels
            headings = self.assign_heading_levels(headings)
            
            # Sort by page number
            headings.sort(key=lambda x: x["page"])
            
            # Format output
            outline = []
            for heading in headings:
                outline.append({
                    "level": heading["level"],
                    "text": heading["text"],
                    "page": heading["page"]
                })
            
            return {
                "title": title,
                "outline": outline
            }
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {"title": "Document", "outline": []}

def process_all_pdfs():
    """Process all PDFs in input directory"""
    input_dir = "input"
    output_dir = "output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = DocumentOutlineExtractor()
    
    # Process each PDF file
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            output_filename = filename.replace('.pdf', '.json')
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"Processing {filename}...")
            
            # Extract outline
            result = extractor.extract_outline(pdf_path)
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Saved outline to {output_filename}")
            print(f"Title: {result['title']}")
            print(f"Found {len(result['outline'])} headings")
            print("-" * 50)

if __name__ == "__main__":
    process_all_pdfs()