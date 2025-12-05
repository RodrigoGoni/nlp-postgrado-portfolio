# 🧠 Portfolio de Procesamiento de Lenguaje Natural

> **Especialización en Procesamiento de Lenguaje Natural**  
> Colección completa de desafíos prácticos implementando técnicas avanzadas de NLP con Deep Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)

---

## 📚 Índice

- [Descripción General](#-descripción-general)
- [Desafíos Implementados](#-desafíos-implementados)
  - [Desafío 1: Bag of Words](#1️⃣-desafío-1-bag-of-words)
  - [Desafío 2: Custom Embeddings](#2️⃣-desafío-2-custom-embeddings)
  - [Desafío 3: Modelo de Lenguaje](#3️⃣-desafío-3-modelo-de-lenguaje)
  - [Desafío 4: Traductor Seq2Seq](#4️⃣-desafío-4-traductor-seq2seq)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Instalación](#-instalación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Resultados Destacados](#-resultados-destacados)
- [Autor](#-autor)

---

## 🎯 Descripción General

Este repositorio consolida los **cuatro desafíos principales** del programa de postgrado en **Procesamiento de Lenguaje Natural**, abarcando desde técnicas clásicas de vectorización hasta arquitecturas neuronales avanzadas como Seq2Seq con atención.

Cada desafío está implementado como un **submódulo Git independiente**, permitiendo navegación modular mientras se mantiene un portfolio unificado para presentación académica y profesional.

### 🎓 Competencias Desarrolladas

- **Técnicas Clásicas de NLP**: TF-IDF, Bag of Words, clasificación por similitud
- **Word Embeddings**: Word2Vec (Skip-gram, CBOW), embeddings personalizados
- **Arquitecturas Recurrentes**: RNN, LSTM, GRU con capas apiladas
- **Modelos de Secuencia**: Encoder-Decoder, Seq2Seq, traducción automática
- **Generación de Texto**: Greedy search, Beam search, Sampling con temperatura
- **Optimización**: Early stopping, Dropout, L2 regularization, Adam optimizer

---

## 🚀 Desafíos Implementados

### 1️⃣ Desafío 1: Bag of Words

**Técnicas clásicas de NLP y clasificación de documentos**

📂 **Repositorio**: [`bag-of-words-npl`](https://github.com/RodrigoGoni/bag-of-words-npl)

#### Descripción
Implementación completa de métodos tradicionales de vectorización y clasificación de textos utilizando el dataset **20 Newsgroups** (20,000 documentos en 20 categorías).

#### Características Principales
- ✅ **Vectorización TF-IDF** para representación de documentos
- ✅ **Clasificación por prototipos** (Zero-shot learning con vecino más cercano)
- ✅ **Modelos Naïve Bayes**: MultinomialNB y ComplementNB optimizados
- ✅ **Análisis de co-ocurrencia** mediante matrices término-documento
- ✅ **Cálculo de similaridad** entre documentos y palabras

#### Resultados
| Modelo | F1-Score Macro |
|--------|---------------|
| Clasificación por Prototipos | 0.5050 |
| MultinomialNB | 0.6833 |
| **ComplementNB** | **0.6950** ⭐ |

#### Stack Tecnológico
`scikit-learn` · `NLTK` · `NumPy` · `Pandas` · `Matplotlib`

---

### 2️⃣ Desafío 2: Custom Embeddings

**Entrenamiento de embeddings personalizados con Word2Vec**

📂 **Repositorio**: [`customs_embeddings`](https://github.com/RodrigoGoni/customs_embeddings)

#### Descripción
Creación de vectores de embeddings desde cero utilizando **Gensim** sobre datasets personalizados: letras de canciones de diversos artistas y textos religiosos (Evangelio de Juan).

#### Características Principales
- ✅ **Word2Vec con Skip-gram y CBOW**
- ✅ **Preprocesamiento con NLTK**: tokenización, stopwords, stemming
- ✅ **Visualización de espacios semánticos** con PCA/t-SNE
- ✅ **Análisis de similitudes** y analogías entre palabras
- ✅ **Web scraping** de fuentes de texto con BeautifulSoup
- ✅ **Exportación a TensorFlow Projector** (vectors.tsv/labels.tsv)

#### Datasets Utilizados
- 📝 Letras de canciones de múltiples artistas (corpus principal)
- 📖 Evangelio de Juan - Biblia de Jerusalén (corpus comparativo)

#### Stack Tecnológico
`Gensim` · `NLTK` · `BeautifulSoup` · `Matplotlib` · `Seaborn`

---

### 3️⃣ Desafío 3: Modelo de Lenguaje

**Generación de texto con redes recurrentes a nivel de carácter**

📂 **Repositorio**: [`chatbot`](https://github.com/RodrigoGoni/chatbot)

#### Descripción
Implementación de modelos de lenguaje basados en RNNs para **generación de texto en español** a nivel de carácter, entrenados sobre un corpus de literatura clásica española del Proyecto Gutenberg (~7.5M caracteres de 8 libros).

#### Características Principales
- ✅ **Arquitecturas implementadas**: SimpleRNN, LSTM, GRU (básicas y avanzadas con capas apiladas)
- ✅ **Estrategias de generación**: Greedy Search, Beam Search, Sampling con temperatura
- ✅ **Anti-overfitting**: Early stopping, Dropout (0.5), Weight Decay (L2)
- ✅ **Métricas de evaluación**: Loss, Accuracy, Perplexity
- ✅ **Análisis lingüístico** con SpaCy (coherencia gramatical)
- ✅ **Gestión con Git LFS** para modelos y datasets grandes

#### Resultados
| Modelo | Val Perplexity | Val Accuracy | Calidad Generación |
|--------|---------------|--------------|-------------------|
| SimpleRNN | 4.23 | 0.52 | Básica |
| GRU | 3.87 | 0.57 | Buena |
| **LSTM** | **3.51** ⭐ | **0.60** ⭐ | **Excelente (mejor estructura gramatical)** |

#### Stack Tecnológico
`TensorFlow/Keras` · `SpaCy` · `NumPy` · `Matplotlib` · `Git LFS`

---

### 4️⃣ Desafío 4: Traductor Seq2Seq

**Traducción automática Inglés→Español con arquitectura Encoder-Decoder**

📂 **Repositorio**: [`seq2seq-translator`](https://github.com/RodrigoGoni/seq2seq-translator)

#### Descripción
Sistema de traducción neuronal basado en **LSTM bidireccionales** con arquitectura sequence-to-sequence, utilizando embeddings pre-entrenados GloVe y optimización inteligente de hiperparámetros.

#### Características Principales
- ✅ **Encoder-Decoder con LSTM** de 2 capas
- ✅ **Embeddings GloVe 50d** para inglés (congelados, ~600k parámetros)
- ✅ **Embeddings entrenables** para español (~1.25M parámetros)
- ✅ **Selección inteligente de hiperparámetros** basada en percentiles del dataset:
  - Vocabulario: Cobertura del 98% (P98)
  - Longitudes de secuencia: Percentil 98
- ✅ **Técnicas de regularización**:
  - Early Stopping (patience=7)
  - Dropout (0.5)
  - L2 Regularization (weight_decay=1e-5)
- ✅ **Dos variantes**: 128 y 256 neuronas LSTM
- ✅ **Guardado/carga de modelos** entrenados (.pth)

#### Dataset
- 📊 **Fuente**: TensorFlow spa-eng
- 📊 **Tamaño**: 118,964 pares de oraciones
- 📊 **Split**: 80% train / 20% validación

#### Arquitectura
```
┌─────────────────────────────────────────┐
│         ENCODER (Inglés)                │
│  Embedding GloVe 50d → LSTM(2 capas)    │
│  Output: Hidden State [2, batch, 128/256]│
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         DECODER (Español)               │
│  Embedding Trainable → LSTM(2 capas)    │
│  FC → Softmax(vocab_size=25k)           │
└─────────────────────────────────────────┘
```

#### Resultados
| Modelo | Parámetros Entrenables | Convergencia |
|--------|----------------------|--------------|
| LSTM-128 | ~4.9M | Early stopping aplicado |
| LSTM-256 | ~4.9M | Mayor capacidad representacional |

#### Stack Tecnológico
`PyTorch` · `torchinfo` · `NumPy` · `Matplotlib` · `GloVe Embeddings`

---

## 🛠️ Tecnologías Utilizadas

### Frameworks de Deep Learning
- **PyTorch** 2.0+ (Desafíos 4)
- **TensorFlow/Keras** 2.0+ (Desafío 3)

### NLP Libraries
- **Gensim** (Word2Vec, embeddings)
- **NLTK** (preprocesamiento)
- **SpaCy** (análisis lingüístico)
- **scikit-learn** (clasificación clásica)

### Visualización & Análisis
- **Matplotlib** / **Seaborn**
- **TensorFlow Projector** (visualización de embeddings)

### Herramientas de Desarrollo
- **Jupyter Notebook** / **VS Code**
- **Git LFS** (gestión de archivos grandes)
- **BeautifulSoup** (web scraping)

---

## 📥 Instalación

### Prerrequisitos
- Python 3.8+
- Git 2.0+
- Git LFS (para el Desafío 3)

### Clonar el Portfolio Completo con Submódulos

```bash
# Clonar repositorio principal
git clone --recurse-submodules https://github.com/RodrigoGoni/nlp-postgrado-portfolio.git
cd nlp-postgrado-portfolio

# Si ya clonaste sin submódulos, inicialízalos:
git submodule update --init --recursive
```

### Instalar Dependencias

#### Opción 1: Entorno virtual unificado (recomendado para exploración)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar todas las dependencias
pip install -r desafio1-bag-of-words/requirements.txt
pip install -r desafio2-custom-embeddings/requirements.txt
pip install -r desafio3-chatbot/requirement.txt
pip install -r desafio4-seq2seq-translator/requirements.txt

# Instalar SpaCy (para Desafío 3)
python -m spacy download es_core_news_sm
```

#### Opción 2: Entornos separados (recomendado para desarrollo)
```bash
# Navegar a cada desafío e instalar individualmente
cd desafio1-bag-of-words
pip install -r requirements.txt
cd ..
# Repetir para cada desafío
```

---

## 📂 Estructura del Proyecto

```
nlp-postgrado-portfolio/
│
├── README.md                           # Este archivo (portfolio principal)
├── .gitignore                          # Exclusiones de Git
├── .gitmodules                         # Configuración de submódulos
├── setup.sh                            # Script de inicialización automática
│
├── desafio1-bag-of-words/              # Submódulo → bag-of-words-npl
│   ├── main.ipynb
│   ├── requirements.txt
│   └── README.md
│
├── desafio2-custom-embeddings/         # Submódulo → customs_embeddings
│   ├── main.ipynb
│   ├── songs_dataset/
│   ├── requirements.txt
│   └── README.md
│
├── desafio3-chatbot/                   # Submódulo → chatbot
│   ├── desafio3.ipynb
│   ├── corpus_espanol.txt (Git LFS)
│   ├── model_*.keras (Git LFS)
│   ├── requirement.txt
│   └── README.md
│
└── desafio4-seq2seq-translator/        # Submódulo → seq2seq-translator
    ├── traductor_simplificado.ipynb
    ├── torch_helpers.py
    └── README.md
```

---

## 📊 Resultados Destacados

### Progresión de Complejidad

| Desafío | Técnica | Nivel | Resultado Clave |
|---------|---------|-------|-----------------|
| 1 | Bag of Words + TF-IDF | Básico | F1=0.695 (ComplementNB) |
| 2 | Word2Vec (Gensim) | Intermedio | Embeddings personalizados coherentes |
| 3 | LSTM a nivel de carácter | Avanzado | Perplexity=3.51, generación coherente |
| 4 | Seq2Seq con GloVe | Experto | Traducción EN→ES con ~4.9M params |

### Métricas Comparativas

```
Clasificación de Textos (Desafío 1):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MultinomialNB    ████████████████░░░░ 68.3%
ComplementNB     ████████████████▓░░░ 69.5% ⭐

Modelos de Lenguaje (Desafío 3):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SimpleRNN        ████████████░░░░░░░░ 52.0%
GRU              ██████████████▓░░░░░ 57.0%
LSTM             ████████████████░░░░ 60.0% ⭐
```

---

## 👤 Autor

**Rodrigo Goñi**

📧 Email: [tu-email@ejemplo.com](mailto:tu-email@ejemplo.com)  
🔗 GitHub: [@RodrigoGoni](https://github.com/RodrigoGoni)  
💼 LinkedIn: [Tu perfil](https://linkedin.com/in/tu-perfil)

---

## 📄 Licencia

Este proyecto está bajo la licencia **Apache 2.0**. Consulta cada submódulo para licencias específicas.

---

## 🙏 Agradecimientos

- **Programa de Postgrado en NLP** - Universidad/Institución
- **Datasets**: 20 Newsgroups, TensorFlow spa-eng, Proyecto Gutenberg
- **Embeddings**: Stanford GloVe
- **Comunidad Open Source**: PyTorch, TensorFlow, Gensim, NLTK, SpaCy

---

## 🔄 Estado del Proyecto

![Estado](https://img.shields.io/badge/Estado-Completo-success)
![Última actualización](https://img.shields.io/badge/Última%20actualización-Diciembre%202025-blue)

**Todos los desafíos están completamente implementados y documentados** ✅

---

<div align="center">

### ⭐ Si este portfolio te resulta útil, considera darle una estrella en GitHub

**Desarrollado con 🧠 por Rodrigo Goñi**

</div>
