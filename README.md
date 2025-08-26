# Adobe Hackathon 2025 – Round 1A  
**Challenge Theme:** Connecting the Dots Through Docs  
**Participant:** Dhruvi Sharma  
**Repository:** [dhruvi-05/adobe_1a](https://github.com/dhruvi-05/adobe_1a)

---

## 🚀 Problem Statement

Your task is to convert a raw PDF document into a structured outline.  
For each PDF, extract:
- Title  
- Headings (H1, H2, H3) with their text and page number

---

## 📝 Sample Output Format (JSON)

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
