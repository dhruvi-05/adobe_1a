Adobe Hackathon 2025 – Round 1A  
Challenge Theme:** Connecting the Dots Through Docs  
Participant: Apoorva Bajpai  
Repository: [dhruvi-05/adobe_1a](https://github.com/dhruvi-05/adobe_1a)

---

🚀 Problem Statement

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

🧠 Approach
-Used PyMuPDF (fitz) to read and analyze PDF structure
-Headings are extracted using a combination of font size, boldness, and position heuristics
-Outputs are generated in the required JSON format

🐳 Docker Setup
-This project runs fully inside a Docker container with:
-CPU only
-Offline (no internet access)
-Supports PDFs up to 50 pages
-Compatible with linux/amd64

🔧 How to Build and Run
🧱 Build the Docker Image
docker build --platform linux/amd64 -t adobe1a-solution:v1 .

▶️ Run the Container
On Unix/Mac:
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none adobe1a-solution:v1

On Windows (PowerShell):
docker run --rm -v C:/Users/apoor/adobe_1a/input:/app/input -v C:/Users/apoor/adobe_1a/output:

📄 Output
For every filename.pdf inside input/, a filename.json is created in output/.
