# 📖 Guía de Instalación y Uso

Esta guía te ayudará a configurar el portfolio completo de desafíos de NLP.

## 📋 Tabla de Contenidos

- [Requisitos Previos](#requisitos-previos)
- [Instalación Rápida](#instalación-rápida)
- [Instalación Manual](#instalación-manual)
- [Uso de los Notebooks](#uso-de-los-notebooks)
- [Solución de Problemas](#solución-de-problemas)

---

## 🔧 Requisitos Previos

### Software Necesario

1. **Python 3.8 o superior**
   ```bash
   python --version  # Debe mostrar 3.8+
   ```

2. **Git 2.0 o superior**
   ```bash
   git --version
   ```

3. **Git LFS** (para archivos grandes del Desafío 3)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # macOS
   brew install git-lfs
   
   # Windows (con Git for Windows incluye Git LFS)
   # O descarga desde: https://git-lfs.github.com/
   
   # Luego inicializa
   git lfs install
   ```

4. **Jupyter Notebook** (recomendado)
   ```bash
   pip install jupyter notebook
   ```

### Hardware Recomendado

- **RAM**: Mínimo 8GB (16GB recomendado para el Desafío 4)
- **Espacio en Disco**: ~5GB para modelos y datasets
- **GPU**: Opcional pero recomendada para entrenamientos (CUDA compatible)

---

## ⚡ Instalación Rápida

### Opción 1: Script Automático (Linux/macOS)

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/nlp-postgrado-portfolio.git
cd nlp-postgrado-portfolio

# 2. Ejecutar el script de configuración
chmod +x setup.sh
./setup.sh

# 3. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate

# Instalar todas las dependencias
pip install -r desafio1-bag-of-words/requirements.txt
pip install -r desafio2-custom-embeddings/requirements.txt
pip install -r desafio3-chatbot/requirement.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # O con CUDA
pip install numpy matplotlib torchinfo

# Descargar modelo de SpaCy
python -m spacy download es_core_news_sm
```

### Opción 2: Manual con Git (Windows/Linux/macOS)

```bash
# 1. Clonar con submódulos
git clone --recurse-submodules https://github.com/TU_USUARIO/nlp-postgrado-portfolio.git
cd nlp-postgrado-portfolio

# 2. Si ya clonaste sin submódulos
git submodule update --init --recursive

# 3. Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Linux/macOS)
source venv/bin/activate

# 4. Instalar dependencias (ver sección siguiente)
```

---

## 🔨 Instalación Manual

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/TU_USUARIO/nlp-postgrado-portfolio.git
cd nlp-postgrado-portfolio
```

### Paso 2: Inicializar Submódulos

```bash
# Añadir submódulos manualmente
git submodule add https://github.com/RodrigoGoni/bag-of-words-npl.git desafio1-bag-of-words
git submodule add https://github.com/RodrigoGoni/customs_embeddings.git desafio2-custom-embeddings
git submodule add https://github.com/RodrigoGoni/chatbot.git desafio3-chatbot
git submodule add https://github.com/RodrigoGoni/seq2seq-translator.git desafio4-seq2seq-translator

# Actualizar submódulos
git submodule update --init --recursive
```

### Paso 3: Crear Entorno Virtual

```bash
# Crear entorno
python -m venv venv

# Activar
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Paso 4: Instalar Dependencias por Desafío

#### Desafío 1: Bag of Words

```bash
cd desafio1-bag-of-words
pip install -r requirements.txt
cd ..
```

**Paquetes principales**: `scikit-learn`, `nltk`, `numpy`, `pandas`, `matplotlib`

#### Desafío 2: Custom Embeddings

```bash
cd desafio2-custom-embeddings
pip install -r requirements.txt
cd ..
```

**Paquetes principales**: `gensim`, `nltk`, `beautifulsoup4`, `matplotlib`, `seaborn`

#### Desafío 3: Chatbot/Modelo de Lenguaje

```bash
cd desafio3-chatbot
pip install -r requirement.txt  # Nota: "requirement.txt" sin 's'

# Descargar modelo de SpaCy para análisis
python -m spacy download es_core_news_sm

# Descargar archivos grandes con Git LFS
git lfs pull
cd ..
```

**Paquetes principales**: `tensorflow`, `keras`, `spacy`, `numpy`, `matplotlib`

#### Desafío 4: Seq2Seq Translator

```bash
cd desafio4-seq2seq-translator

# Instalar PyTorch (elige según tu sistema)
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Con CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Con CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Otros paquetes
pip install numpy matplotlib torchinfo
cd ..
```

**Paquetes principales**: `pytorch`, `torchinfo`, `numpy`, `matplotlib`

### Paso 5: Verificar Instalación

```bash
# Verificar Python
python --version

# Verificar paquetes críticos
python -c "import sklearn; import gensim; import tensorflow; import torch; print('✅ Todo OK')"
```

---

## 🚀 Uso de los Notebooks

### Iniciar Jupyter Notebook

```bash
# Desde el directorio raíz del portfolio
jupyter notebook
```

O si prefieres JupyterLab:

```bash
pip install jupyterlab
jupyter lab
```

### Ejecutar Notebooks por Desafío

#### Desafío 1: Bag of Words

```bash
cd desafio1-bag-of-words
jupyter notebook main.ipynb
```

**Contenido**:
- Vectorización TF-IDF
- Clasificación con Naïve Bayes
- Análisis de similaridad

#### Desafío 2: Custom Embeddings

```bash
cd desafio2-custom-embeddings
jupyter notebook main.ipynb
```

**Contenido**:
- Entrenamiento Word2Vec
- Visualización de embeddings
- Análisis de similitudes

#### Desafío 3: Modelo de Lenguaje

```bash
cd desafio3-chatbot
jupyter notebook desafio3.ipynb
```

**Contenido**:
- Entrenamiento RNN/LSTM/GRU
- Generación de texto
- Análisis de calidad con SpaCy

#### Desafío 4: Seq2Seq Translator

```bash
cd desafio4-seq2seq-translator
jupyter notebook traductor_simplificado.ipynb
```

**Contenido**:
- Encoder-Decoder LSTM
- Embeddings GloVe
- Traducción EN→ES

---

## 🐛 Solución de Problemas

### Problema: Submódulos vacíos

```bash
# Solución
git submodule update --init --recursive
```

### Problema: Git LFS no descarga archivos

```bash
# Instalar Git LFS
git lfs install

# Descargar archivos
cd desafio3-chatbot
git lfs pull
```

### Problema: Falta modelo de SpaCy

```bash
# Error: Can't find model 'es_core_news_sm'
python -m spacy download es_core_news_sm
```

### Problema: PyTorch no encuentra CUDA

```bash
# Verificar CUDA disponible
python -c "import torch; print(torch.cuda.is_available())"

# Si es False pero tienes GPU, reinstala PyTorch con CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problema: Out of Memory (OOM) en notebooks

**Solución temporal**: Reducir batch size o tamaño de modelo en las celdas de configuración.

```python
# Ejemplo en el Desafío 4
batch_size = 32  # Reducir a 16 o 8
hidden_size = 128  # Reducir a 64
```

### Problema: NLTK data no encontrado

```bash
# Descargar datos NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Problema: Jupyter Kernel no inicia

```bash
# Reinstalar ipykernel
pip install --upgrade ipykernel
python -m ipykernel install --user --name=venv
```

---

## 📚 Recursos Adicionales

### Datasets

Los datasets se descargan automáticamente en los notebooks, pero también puedes obtenerlos manualmente:

- **20 Newsgroups**: `sklearn.datasets.fetch_20newsgroups()`
- **TensorFlow spa-eng**: http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
- **GloVe Embeddings**: https://nlp.stanford.edu/projects/glove/

### Documentación

- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Docs](https://www.tensorflow.org/api_docs)
- [Gensim Docs](https://radimrehurek.com/gensim/)
- [SpaCy Docs](https://spacy.io/api)

---

## 🆘 Soporte

Si encuentras problemas:

1. Revisa esta guía de solución de problemas
2. Consulta el README de cada desafío individual
3. Abre un issue en GitHub: https://github.com/TU_USUARIO/nlp-postgrado-portfolio/issues

---

## ✅ Checklist de Instalación

- [ ] Python 3.8+ instalado
- [ ] Git instalado
- [ ] Git LFS instalado y configurado
- [ ] Repositorio clonado con submódulos
- [ ] Entorno virtual creado y activado
- [ ] Dependencias del Desafío 1 instaladas
- [ ] Dependencias del Desafío 2 instaladas
- [ ] Dependencias del Desafío 3 instaladas + modelo SpaCy
- [ ] Dependencias del Desafío 4 instaladas (PyTorch)
- [ ] Jupyter Notebook funcional
- [ ] Notebooks se abren sin errores

---

**¡Listo para explorar NLP! 🚀**
