# 🩺 MedicalGPT

A powerful clinical decision support system that leverages RAG (Retrieval Augmented Generation) to provide diagnostic insights based on patient cases.

## 🌟 Features

- **Clinical Query Interface**: Simple and intuitive interface for entering medical queries
- **Case Retrieval**: Efficiently retrieves relevant patient cases using FAISS similarity search
- **AI-Powered Analysis**: Generates diagnostic insights using state-of-the-art language models
- **Modern UI**: Clean, responsive interface built with Streamlit
- **Context Display**: Expandable view of retrieved clinical notes for transparency

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MedicalGPT.git
cd MedicalGPT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Launch the application using Streamlit:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` by default.

## 📦 Project Structure

```
MedicalGPT/
├── app.py              # Main Streamlit application
├── assets/            # Static assets
├── data/             # Clinical data and embeddings
├── src/              # Source code
│   ├── retriever.py  # Document retrieval logic
│   └── generator.py  # Text generation components
└── requirements.txt   # Project dependencies
```

## 🛠️ Technologies Used

- **Streamlit**: Web interface and application framework
- **FAISS**: Efficient similarity search and clustering
- **Sentence Transformers**: Text embeddings for retrieval
- **PyTorch**: Deep learning framework
- **Transformers**: NLP models for text generation
- **Pandas**: Data manipulation and analysis

## 💡 Usage

1. Enter a clinical query in the text input field
2. The system will retrieve relevant patient cases
3. Review the retrieved clinical notes in the expandable sections
4. Examine the AI-generated diagnostic insights

## 📝 Note

This tool is designed to assist medical professionals and should not be used as a replacement for professional medical judgment. Always consult with qualified healthcare providers for medical decisions.