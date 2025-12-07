import json
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import pandas as pd
import numpy as np


STEMMER = RSLPStemmer()
STOPWORDS_PORTUGUES = set(stopwords.words('portuguese'))


COLECAO_DOCUMENTOS = {}
VOCABULARIO = set()  
INDICE_PROXIMO_DOC = 0
DADOS_JSON = None


MATRIZ_TFIDF = None
INDICE_INVERTIDO = None


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
    texto = re.sub(r'[^a-zà-ú\s]', '', texto) 
    return texto

def preprocessar_documento(texto, stemmer, stopwords_portugues):
    texto_limpo = limpar_texto(texto)
    tokens = texto_limpo.split()
    tokens_filtrados = [stemmer.stem(t) for t in tokens if t not in stopwords_portugues]
    return tokens_filtrados



def construir_matriz_boolean(docs_processados, nomes_docs):
    vocabulario = sorted(set([termo for doc in docs_processados for termo in doc]))
    matriz = pd.DataFrame(0, index=vocabulario, columns=nomes_docs)
    for nome, termos in zip(nomes_docs, docs_processados):
        for termo in termos:
            matriz.loc[termo, nome] = 1
    return matriz

def calcular_tf(docs_processados, nomes_docs):
    vocabulario = sorted(set([t for doc in docs_processados for t in doc]))
    matriz_tf = pd.DataFrame(0.0, index=vocabulario, columns=nomes_docs)
    for nome, termos in zip(nomes_docs, docs_processados):
        for termo in termos:
            matriz_tf.loc[termo, nome] += 1
            
    
    matriz_tf = matriz_tf.map(lambda x: 1 + np.log2(x) if x > 0 else 0)
    return matriz_tf

def calcular_idf(matriz_tf):
    N = matriz_tf.shape[1]
    ni = (matriz_tf > 0).sum(axis=1)
    idf = np.log2(N / ni)
    return pd.Series(idf, index=matriz_tf.index, name="IDF")

def calcular_tfidf(matriz_tf, idf):
    matriz_tfidf = matriz_tf.mul(idf, axis=0)
    return matriz_tfidf

def construir_indice_invertido(docs_tokens):
    indice = {}
    for doc, tokens in docs_tokens.items():
        for t in tokens:
            if t not in indice:
                indice[t] = set()
            indice[t].add(doc)
    return indice


def busca_boolean(indice, consulta):
    consulta_original = consulta.lower()
    consulta_formatada = consulta_original.replace(' and not ', ' AND NOT ').replace(' and ', ' AND ').replace(' or ', ' OR ')
    termos_consulta = consulta_formatada.split()

   
    def stem_e_filtra(tokens):
        return [STEMMER.stem(t) for t in tokens if t not in STOPWORDS_PORTUGUES]
    
    if 'AND NOT' in termos_consulta:
        try:
            op_index = termos_consulta.index('AND NOT')
            termo1_stem = stem_e_filtra([termos_consulta[op_index - 1]])[0]
            termo2_stem = stem_e_filtra([termos_consulta[op_index + 1]])[0]
            set1 = indice.get(termo1_stem, set())
            set2 = indice.get(termo2_stem, set())
            return set1 - set2
        except (IndexError, TypeError):
            print("Erro na sintaxe 'AND NOT'. Use: 'termo1 AND NOT termo2'")
    elif 'AND' in termos_consulta:
        try:
            op_index = termos_consulta.index('AND')
            termo1_stem = stem_e_filtra([termos_consulta[op_index - 1]])[0]
            termo2_stem = stem_e_filtra([termos_consulta[op_index + 1]])[0]
            set1 = indice.get(termo1_stem, set())
            set2 = indice.get(termo2_stem, set())
            return set1 & set2
        except (IndexError, TypeError):
            print("Erro na sintaxe 'AND'. Use: 'termo1 AND termo2'")
            return set()

    elif 'OR' in termos_consulta:
        try:
            op_index = termos_consulta.index('OR')
            termo1_stem = stem_e_filtra([termos_consulta[op_index - 1]])[0]
            termo2_stem = stem_e_filtra([termos_consulta[op_index + 1]])[0]
            set1 = indice.get(termo1_stem, set())
            set2 = indice.get(termo2_stem, set())
            return set1 | set2
        except (IndexError, TypeError):
            print("Erro na sintaxe 'OR'. Use: 'termo1 OR termo2'")
            return set()
    else:
        termos_processados = stem_e_filtra(termos_consulta)
        if termos_processados:
            return indice.get(termos_processados[0], set())
        return set()


def atualizar_estruturas_cache():

    global MATRIZ_TFIDF
    global INDICE_INVERTIDO

    if not COLECAO_DOCUMENTOS:
        MATRIZ_TFIDF = None
        INDICE_INVERTIDO = None
        print("[INFO] Estruturas de cache limpas (coleção vazia).")
        return

    nomes_docs = list(COLECAO_DOCUMENTOS.keys())
    docs_processados = []
    docs_tokens_map = {} 
    
    
    for doc_id, conteudo in COLECAO_DOCUMENTOS.items():
        tokens = preprocessar_documento(conteudo, STEMMER, STOPWORDS_PORTUGUES)
        docs_processados.append(tokens)
        docs_tokens_map[doc_id] = tokens
    
   
    matriz_tf = calcular_tf(docs_processados, nomes_docs)
    idf = calcular_idf(matriz_tf)
    MATRIZ_TFIDF = calcular_tfidf(matriz_tf, idf)
    
  
    INDICE_INVERTIDO = construir_indice_invertido(docs_tokens_map)
    

def add_all_doc():
    dados = ler_json('colecao - trabalho 01.json')

    if dados:
        docs_adicionados = 0
        for doc in dados:
            doc_id = doc.get("name")
            conteudo = doc.get("content")

            if doc_id is not None and conteudo is not None:
                COLECAO_DOCUMENTOS[doc_id] = conteudo
                tokens = preprocessar_documento(conteudo, STEMMER, STOPWORDS_PORTUGUES)
                VOCABULARIO.update(tokens)
                docs_adicionados += 1
        
        if docs_adicionados > 0:
            print(f"Total de {docs_adicionados} documentos adicionados à coleção.")
            atualizar_estruturas_cache() 
        else:
            print("Nenhum documento novo adicionado.")

def add_docadoc():
    global INDICE_PROXIMO_DOC
    global DADOS_JSON
    global VOCABULARIO

    if DADOS_JSON is None:
        DADOS_JSON = ler_json('colecao - trabalho 01.json')
    
    if DADOS_JSON and INDICE_PROXIMO_DOC < len(DADOS_JSON):
        doc = DADOS_JSON[INDICE_PROXIMO_DOC]
        doc_id = doc.get("name")
        conteudo = doc.get("content")

        if doc_id and conteudo:
            COLECAO_DOCUMENTOS[doc_id] = conteudo
            tokens = preprocessar_documento(conteudo, STEMMER, STOPWORDS_PORTUGUES)
            VOCABULARIO.update(tokens)
            print(f"Documento '{doc_id}' adicionado à coleção.")
            INDICE_PROXIMO_DOC += 1
            
            atualizar_estruturas_cache() 
    else:
        print("Todos os documentos já foram adicionados à coleção ou o arquivo JSON está vazio.")


def remover_documento(doc_id):
    global VOCABULARIO
    
    if doc_id not in COLECAO_DOCUMENTOS:
        print(f"Erro: O documento com ID '{doc_id}' não existe na coleção.")
        return None
    del COLECAO_DOCUMENTOS[doc_id]
    print(f"Documento '{doc_id}' removido com sucesso.")
    VOCABULARIO_NOVO = set()
    for conteudo in COLECAO_DOCUMENTOS.values():
        tokens = preprocessar_documento(conteudo, STEMMER, STOPWORDS_PORTUGUES)
        VOCABULARIO_NOVO.update(tokens)

    VOCABULARIO = VOCABULARIO_NOVO
    print("Vocabulário atualizado após remoção.")

    atualizar_estruturas_cache() 



def main():
    while True: 
        print("\n--- MENU DE OPÇÕES ---")
        print("1 - Adicionar um documento por vez à coleção")
        print("2 - Adicionar todos os documentos da lista.")
        print("3 - Remover um documento da coleção pelo seu identificador.")
        print("4 - Exibir o vocabulário atualizado.")
        print("5 - Exibir a matriz TF-IDF atual (CACHE).")
        print("6 - Exibir o índice invertido completo (CACHE).")
        print("7 - Realizar consultas booleanas.")
        print("8 - Realizar consultas por similaridade.")
        print("9 - Realizar consultas por frase.")
        print("10 - Sair.")
        
        try:
            res = int(input("Digite um número: "))
        except ValueError:
            print("Entrada inválida. Digite um número de 1 a 10.")
            continue
            
        if res == 10:
            print("Saindo do programa.")
            break
            
        match res:
            case 1:
                add_docadoc()
            case 2:
                add_all_doc()
            case 3:
                doc_id = input("Digite o identificador do documento a remover: ")
                remover_documento(doc_id)
            case 4:
                print("\nVocabulário Atualizado\n")
                if not VOCABULARIO:
                    print("O vocabulário está vazio. Adicione documentos primeiro.")
                else:
                    vocabulario_ordenado = sorted(list(VOCABULARIO))
                    print(vocabulario_ordenado)
            case 5:
                if MATRIZ_TFIDF is None:
                    print("Não há documentos na coleção ou a matriz ainda não foi calculada.")
                else:
                    print("\nMatriz TF-IDF\n")
                    print(MATRIZ_TFIDF)
            case 6:
                if INDICE_INVERTIDO is None:
                    print("Não há documentos na coleção ou o índice ainda não foi calculado.")
                else:
                    print("\nÍndice Invertido (Termo: {IDs de Documentos})\n")
                    for termo in sorted(INDICE_INVERTIDO.keys()):
                        documentos = ', '.join(sorted(list(INDICE_INVERTIDO[termo])))
                        print(f"{termo}: {{{documentos}}}")
            case 7:
                if INDICE_INVERTIDO is None:
                    print("ERRO: O Índice Invertido ainda não foi calculado. Adicione documentos primeiro.")
                else:
                    consulta_usuario = input("Digite a consulta booleana (ex: 'termo1 AND termo2' ou 'termo'): ")
                    resultados = busca_boolean(INDICE_INVERTIDO, consulta_usuario)
                    
                    print("\nResultado da Busca Booleana")
                    
                    if resultados:
                        docs_encontrados = sorted(list(resultados))
                        print(f"Documentos Encontrados ({len(docs_encontrados)}): {', '.join(docs_encontrados)}")
                    else:
                        print("Nenhum documento encontrado para esta consulta.")
                break
            case 8:
                print("Implementação da Consulta por Similaridade pendente.")
            case 9:
                print("Implementação da Consulta por Frase pendente.")
            case _:
                print("Opção inválida.")


if __name__ == "__main__":
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Baixando stopwords do NLTK...")
        nltk.download('stopwords')
        
    main()