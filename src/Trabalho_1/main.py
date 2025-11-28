import json
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

COLECAO_DOCUMENTOS = {}
VOCABULARIO = {}

def ler_jason(nome_arquivo):
    caminho_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_arquivo = os.path.join(caminho_base, 'data', nome_arquivo)

    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        return dados
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Erro: Não foi possível decodificar o arquivo JSON em '{caminho_arquivo}'.")
        return None

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-zà-úÀ-Ú\s]', '', texto)
    return texto

def preprocessar_documento(texto, stemmer, stopwords_portugues):
    texto_limpo = limpar_texto(texto)
    tokens = texto_limpo.split()
    tokens_filtrados = [stemmer.stem(t) for t in tokens if t not in stopwords_portugues]
    return tokens_filtrados


def add_all_doc():
    dados = ler_jason('colecao - trabalho 01.json')

    if dados:
        for doc in dados:
            doc_id = doc.get("name")
            conteudo = doc.get("content")

            if doc_id is not None and conteudo is not None:
                COLECAO_DOCUMENTOS[doc_id] = conteudo
                tokens = preprocessar_documento(conteudo)
                VOCABULARIO.update(tokens)

def main():
    while 1:
        print("1 - Adicionar um documento por vez à coleção")
        print("2 - Adicionar todos os documentos da lista.")
        print("3 - Remover um documento da coleção pelo seu identificador.")
        print("4 - Exibir o vocabulário atualizado.")
        print("5 - Exibir a matriz TF-IDF atual.")
        print("6 - Exibir o índice invertido completo por posição de palavras.")
        print("7 - Realizar consultas booleanas.")
        print("8 - Realizar consultas por similaridade.")
        print("9 - Realizar consultas por frase.")

if __name__ == "__main__":
    main()