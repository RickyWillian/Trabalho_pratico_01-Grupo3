import json
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

STEMMER = RSLPStemmer()
STOPWORDS_PORTUGUES = set(stopwords.words('portuguese'))

COLECAO_DOCUMENTOS = {}
VOCABULARIO = set()  #evita duplicação automatica no vocabulario 
INDICE_PROXIMO_DOC = 0
DADOS_JSON = None

def ler_json(nome_arquivo):
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
    dados = ler_json('colecao - trabalho 01.json')

    if dados:
        for doc in dados:
            doc_id = doc.get("name")
            conteudo = doc.get("content")

            if doc_id is not None and conteudo is not None:
                COLECAO_DOCUMENTOS[doc_id] = conteudo
                tokens = preprocessar_documento(conteudo,STEMMER, STOPWORDS_PORTUGUES)
                VOCABULARIO.update(tokens)

def add_docadoc():
    global INDICE_PROXIMO_DOC
    global DADOS_JSON
    global VOCABULARIO

    if DADOS_JSON is None:
        DADOS_JSON = ler_json('colecao - trabalho 01.json')
        if INDICE_PROXIMO_DOC < len(DADOS_JSON):
            doc = DADOS_JSON[INDICE_PROXIMO_DOC]
            doc_id = doc.get("name")
            conteudo = doc.get("content")

        if doc and conteudo:
            COLECAO_DOCUMENTOS[doc_id] = conteudo
            tokens = preprocessar_documento(conteudo, STEMMER, STOPWORDS_PORTUGUES)
            VOCABULARIO.update(tokens)
            print(f"Documento '{doc_id}' adicionado à coleção.")
            INDICE_PROXIMO_DOC += 1
    else:
        print("Todos os documentos ja foram adicionados à coleção")

def remover_documento(doc_id):
    
    if doc_id not in COLECAO_DOCUMENTOS:
        print(f"Erro: O documento com ID '{doc_id}' não existe na coleção.")
        return  

    # Remover o documento
    del COLECAO_DOCUMENTOS[doc_id]
    print(f"Documento '{doc_id}' removido com sucesso.")

   
    global VOCABULARIO
    VOCABULARIO_NOVO = set()

    for conteudo in COLECAO_DOCUMENTOS.values():
        tokens = preprocessar_documento(conteudo, STEMMER, STOPWORDS_PORTUGUES)
        VOCABULARIO_NOVO.update(tokens)

    VOCABULARIO = VOCABULARIO_NOVO
    print("Vocabulário atualizado após remoção.")


def main():
    while 1:
        print("Digite um numero de 1 a 10 para as opções do menu:\n")
        print("1 - Adicionar um documento por vez à coleção")
        print("2 - Adicionar todos os documentos da lista.")
        print("3 - Remover um documento da coleção pelo seu identificador.")
        print("4 - Exibir o vocabulário atualizado.")
        print("5 - Exibir a matriz TF-IDF atual.")
        print("6 - Exibir o índice invertido completo por posição de palavras.")
        print("7 - Realizar consultas booleanas.")
        print("8 - Realizar consultas por similaridade.")
        print("9 - Realizar consultas por frase.")
        print("10 - Sair.")
        res = int(input())
        if(res==10):
            break
        # Tentando organizar o menu para criar as respectivas funções
        match res:
            case 1:
                add_docadoc()
                break
            case 2:
                add_all_doc()
                break
            case 3:
                doc_id = input("Digite o identificador do documento a remover: ")
                remover_documento(doc_id)
                break
            case 4:
                break
            case 5:
                break
            case 6:
                break
            case 7:
                break
            case 8:
                break
            case 9:
                break


if __name__ == "__main__":
    main()