# 🎓 Guía de Presentación para Profesores

## 📋 Resumen Ejecutivo

Este portfolio demuestra dominio completo de técnicas de **Procesamiento de Lenguaje Natural**, desde métodos clásicos hasta arquitecturas neuronales avanzadas, con implementaciones funcionales y resultados medibles en cada desafío.

---

## 🎯 Competencias Demostradas

### 1️⃣ Fundamentos de NLP Clásico
- ✅ Vectorización de texto (TF-IDF, Bag of Words)
- ✅ Clasificación supervisada (Naïve Bayes optimizado)
- ✅ Métricas de evaluación (F1-Score, Precision, Recall)
- ✅ Análisis de co-ocurrencia y similaridad

### 2️⃣ Word Embeddings
- ✅ Entrenamiento de Word2Vec desde cero
- ✅ Preprocesamiento de corpus (tokenización, limpieza)
- ✅ Análisis de espacios semánticos
- ✅ Visualización de embeddings (t-SNE, PCA)

### 3️⃣ Redes Neuronales Recurrentes
- ✅ Implementación de RNN, LSTM, GRU
- ✅ Modelos de lenguaje a nivel de carácter
- ✅ Estrategias de generación (Greedy, Beam Search, Sampling)
- ✅ Técnicas anti-overfitting (Early Stopping, Dropout, L2)

### 4️⃣ Arquitecturas Seq2Seq
- ✅ Encoder-Decoder con LSTM
- ✅ Transfer learning con embeddings pre-entrenados (GloVe)
- ✅ Optimización de hiperparámetros basada en datos
- ✅ Traducción automática (inglés→español)

---

## 📊 Resultados Cuantificables

| Desafío | Métrica Principal | Resultado | Estado |
|---------|------------------|-----------|--------|
| 1 - Bag of Words | F1-Score (ComplementNB) | **0.6950** | ✅ Superó baseline |
| 2 - Embeddings | Coherencia semántica | **Alta** | ✅ Analogías válidas |
| 3 - LSTM | Perplexity / Accuracy | **3.51 / 0.60** | ✅ Generación coherente |
| 4 - Seq2Seq | Convergencia | **Early stop aplicado** | ✅ Modelo funcional |

---

## 🛠️ Stack Tecnológico

### Frameworks & Libraries
- **Deep Learning**: PyTorch, TensorFlow/Keras
- **NLP**: Gensim, NLTK, SpaCy
- **ML Clásico**: scikit-learn
- **Visualización**: Matplotlib, Seaborn
- **Gestión**: Git, Git LFS, Jupyter

### Buenas Prácticas Aplicadas
- ✅ Control de versiones con Git (submódulos)
- ✅ Documentación completa (READMEs detallados)
- ✅ Código modular y reutilizable
- ✅ Gestión de archivos grandes (Git LFS)
- ✅ Reproducibilidad (requirements.txt, seeds aleatorias)

---

## 🔍 Navegación del Portfolio

### Repositorio Principal
https://github.com/RodrigoGoni/nlp-postgrado-portfolio

### Repositorios Individuales (Submódulos)

1. **Desafío 1**: https://github.com/RodrigoGoni/bag-of-words-npl
   - Notebook: `main.ipynb`
   - Técnicas: TF-IDF, Naïve Bayes, similaridad

2. **Desafío 2**: https://github.com/RodrigoGoni/customs_embeddings
   - Notebook: `main.ipynb`
   - Técnicas: Word2Vec, embeddings personalizados

3. **Desafío 3**: https://github.com/RodrigoGoni/chatbot
   - Notebook: `desafio3.ipynb`
   - Técnicas: RNN/LSTM/GRU, generación de texto

4. **Desafío 4**: https://github.com/RodrigoGoni/seq2seq-translator
   - Notebook: `traductor_simplificado.ipynb`
   - Técnicas: Encoder-Decoder, GloVe, traducción

---

## 💡 Aspectos Destacados

### Originalidad
- **Datasets personalizados** en Desafío 2 (letras de canciones)
- **Optimización inteligente** en Desafío 4 (hiperparámetros basados en P98)
- **Análisis lingüístico** en Desafío 3 (evaluación con SpaCy)

### Complejidad Técnica
- **~4.9M parámetros entrenables** en Seq2Seq (Desafío 4)
- **Corpus de 7.5M caracteres** procesado (Desafío 3)
- **118K pares de oraciones** para traducción (Desafío 4)

### Documentación
- **4 READMEs detallados** (uno por desafío)
- **1 README principal** unificado
- **Guías de instalación** paso a paso
- **Métricas y resultados** claramente presentados

---

## 🚀 Ejecución Rápida (Para Revisión)

```bash
# Clonar portfolio completo
git clone --recurse-submodules https://github.com/RodrigoGoni/nlp-postgrado-portfolio.git
cd nlp-postgrado-portfolio

# Instalar dependencias (opción rápida)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
chmod +x setup.sh && ./setup.sh

# Abrir notebooks
jupyter notebook
```

### Notebooks Clave para Revisión

1. **Desafío 1**: `desafio1-bag-of-words/main.ipynb` → TF-IDF y clasificación
2. **Desafío 2**: `desafio2-custom-embeddings/main.ipynb` → Word2Vec custom
3. **Desafío 3**: `desafio3-chatbot/desafio3.ipynb` → LSTM generativo
4. **Desafío 4**: `desafio4-seq2seq-translator/traductor_simplificado.ipynb` → Traductor

**Tiempo estimado de revisión**: 15-20 minutos por desafío (notebooks pre-ejecutados)

---

## 📈 Progresión Pedagógica

El portfolio sigue una progresión lógica de complejidad:

```
Fundamentos Clásicos → Embeddings → RNNs → Seq2Seq
     (Desafío 1)    →  (Desafío 2) → (Desafío 3) → (Desafío 4)
```

### Conexiones Entre Desafíos

- **1→2**: De vectores dispersos (TF-IDF) a vectores densos (embeddings)
- **2→3**: De embeddings estáticos a representaciones contextuales (RNN)
- **3→4**: De modelado de lenguaje a traducción (secuencia a secuencia)

---

## 🎓 Reflexión Crítica

### Fortalezas
- ✅ Implementaciones completas y funcionales
- ✅ Experimentación con múltiples arquitecturas
- ✅ Análisis cuantitativo de resultados
- ✅ Código limpio y bien documentado

### Áreas de Mejora Futuras
- 🔄 Implementar mecanismo de atención en Seq2Seq
- 🔄 Explorar Transformers (BERT, GPT)
- 🔄 Fine-tuning de modelos pre-entrenados
- 🔄 Despliegue en producción (API REST)

---

## 📞 Contacto

**Rodrigo Goñi**  
📧 Email: [tu-email]  
🔗 GitHub: [@RodrigoGoni](https://github.com/RodrigoGoni)

---

## ✅ Checklist de Evaluación

Para facilitar la revisión, los desafíos cumplen con:

- [x] Implementación completa y funcional
- [x] Código ejecutable sin errores
- [x] Documentación clara (README + comentarios)
- [x] Resultados cuantitativos reportados
- [x] Buenas prácticas de ingeniería de software
- [x] Reproducibilidad (requirements.txt, seeds)
- [x] Visualizaciones de resultados
- [x] Análisis de métricas de evaluación

---

**Nota**: Todos los notebooks están pre-ejecutados con outputs guardados para facilitar la revisión sin necesidad de re-entrenar modelos (lo cual puede tomar horas).

---

<div align="center">

### 🎓 Portfolio desarrollado para el Programa de Postgrado en NLP

**Diciembre 2025**

</div>
