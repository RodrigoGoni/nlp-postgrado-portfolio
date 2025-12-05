#!/bin/bash

# ============================================
# Script de Inicialización del Portfolio NLP
# ============================================
# Autor: Rodrigo Goñi
# Descripción: Configura automáticamente todos los submódulos del portfolio

set -e  # Salir si hay errores

echo "🚀 Inicializando Portfolio de NLP - Postgrado"
echo "=============================================="
echo ""

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar que estamos en un repositorio git
if [ ! -d .git ]; then
    echo -e "${YELLOW}⚠️  Inicializando repositorio Git...${NC}"
    git init
    echo -e "${GREEN}✅ Repositorio Git creado${NC}"
fi

echo -e "${BLUE}📦 Paso 1: Añadiendo submódulos${NC}"
echo "-------------------------------------------"

# Añadir submódulos si no existen
if [ ! -d "desafio1-bag-of-words" ]; then
    echo "Añadiendo Desafío 1: Bag of Words..."
    git submodule add https://github.com/RodrigoGoni/bag-of-words-npl.git desafio1-bag-of-words
    echo -e "${GREEN}✅ Desafío 1 añadido${NC}"
else
    echo -e "${GREEN}✅ Desafío 1 ya existe${NC}"
fi

if [ ! -d "desafio2-custom-embeddings" ]; then
    echo "Añadiendo Desafío 2: Custom Embeddings..."
    git submodule add https://github.com/RodrigoGoni/customs_embeddings.git desafio2-custom-embeddings
    echo -e "${GREEN}✅ Desafío 2 añadido${NC}"
else
    echo -e "${GREEN}✅ Desafío 2 ya existe${NC}"
fi

if [ ! -d "desafio3-chatbot" ]; then
    echo "Añadiendo Desafío 3: Chatbot..."
    git submodule add https://github.com/RodrigoGoni/chatbot.git desafio3-chatbot
    echo -e "${GREEN}✅ Desafío 3 añadido${NC}"
else
    echo -e "${GREEN}✅ Desafío 3 ya existe${NC}"
fi

if [ ! -d "desafio4-seq2seq-translator" ]; then
    echo "Añadiendo Desafío 4: Seq2Seq Translator..."
    git submodule add https://github.com/RodrigoGoni/seq2seq-translator.git desafio4-seq2seq-translator
    echo -e "${GREEN}✅ Desafío 4 añadido${NC}"
else
    echo -e "${GREEN}✅ Desafío 4 ya existe${NC}"
fi

echo ""
echo -e "${BLUE}🔄 Paso 2: Inicializando y actualizando submódulos${NC}"
echo "-------------------------------------------"
git submodule update --init --recursive
echo -e "${GREEN}✅ Submódulos actualizados${NC}"

echo ""
echo -e "${BLUE}🔍 Paso 3: Verificando estructura${NC}"
echo "-------------------------------------------"
echo "Estructura del portfolio:"
tree -L 2 -d 2>/dev/null || ls -R | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/   /' -e 's/-/|/'

echo ""
echo -e "${BLUE}📋 Paso 4: Verificando Git LFS (Desafío 3)${NC}"
echo "-------------------------------------------"
if command -v git-lfs &> /dev/null; then
    echo -e "${GREEN}✅ Git LFS está instalado${NC}"
    cd desafio3-chatbot 2>/dev/null && git lfs pull && cd .. || echo -e "${YELLOW}⚠️  No se pudo acceder al Desafío 3${NC}"
else
    echo -e "${YELLOW}⚠️  Git LFS no está instalado. Instálalo para descargar modelos grandes:${NC}"
    echo "   Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "   macOS: brew install git-lfs"
    echo "   Luego ejecuta: git lfs install"
fi

echo ""
echo -e "${GREEN}=============================================="
echo "✅ ¡Inicialización completada exitosamente!"
echo "===============================================${NC}"
echo ""
echo -e "${BLUE}📚 Próximos pasos:${NC}"
echo ""
echo "1. Crear entorno virtual:"
echo "   python -m venv venv"
echo "   source venv/bin/activate  # En Windows: venv\\Scripts\\activate"
echo ""
echo "2. Instalar dependencias (elige una opción):"
echo ""
echo "   Opción A - Instalar todo:"
echo "   for dir in desafio*; do"
echo "       pip install -r \$dir/requirements.txt 2>/dev/null || pip install -r \$dir/requirement.txt 2>/dev/null"
echo "   done"
echo ""
echo "   Opción B - Instalar por desafío:"
echo "   cd desafio1-bag-of-words && pip install -r requirements.txt"
echo ""
echo "3. Descargar modelo de SpaCy (para Desafío 3):"
echo "   python -m spacy download es_core_news_sm"
echo ""
echo "4. Abrir notebooks:"
echo "   jupyter notebook"
echo ""
echo -e "${YELLOW}📖 Consulta el README.md principal para más detalles${NC}"
echo ""
