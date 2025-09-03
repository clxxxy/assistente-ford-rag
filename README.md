<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Ford_logo_flat.svg/1000px-Ford_logo_flat.svg.png' width='150'/>

# Assistente de Manual Ford — Pipeline RAG
Uma aplicação com interface Streamlit que permite carregar um PDF, indexá-lo em um Vector Store Chroma com embeddings HuggingFace, e consultar via SLM local (Ollama) com recuperação (RAG).

## Sumário

1. [Visão Geral](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#vis%C3%A3o-geral)
2. [Arquitetura](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#arquitetura)
3. [Decisões de Design](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#decis%C3%B5es-de-design)
4. [Engenharia de Prompt](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#engenharia-de-prompt)
5. [Pipeline Detalhada](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#pipeline-detalhada-indexa%C3%A7%C3%A3o--chunkeriza%C3%A7%C3%A3o)
6. [Embeddings](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#embeddings)
7. [SLM](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#slm)
8. [Como Executar a Aplicação](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#como-executar-a-aplica%C3%A7%C3%A3o)
9. [Como Usar a Aplicação](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#como-usar-a-aplica%C3%A7%C3%A3o)
10. [Limitações Conhecidas e Próximos Passos](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#limita%C3%A7%C3%B5es-conhecidas-e-pr%C3%B3ximos-passos)

- - -

## Visão geral

Este projeto demonstra um RAG para perguntas e respostas: o usuário o manual do veículo do veículo Ford em PDF, o sistema fragmenta o conteúdo (chunks), gera embeddings e salva em um ChromaDB, uma cadeia de recuperação consulta os trechos mais relevantes e monta um prompt para o SLM responder com as fontes.

- Frontend: Streamlit com UI para upload, progresso de indexação e histórico de Q&A. 
- Indexação: PyPDFLoader → RecursiveCharacterTextSplitter → HuggingFaceEmbeddings → Chroma. 
- Recuperação: Retriever com MMR para equilíbrio entre relevância e diversidade.
- SLM local: qwen2:1.5b via Ollama (local).

> O vídeo de demonstração da aplicação está presente [neste link](https://mega.nz/file/jLhAGbwR#-QAMepK4pHb-4bTUJ1rMj0EKIIyTrUjbINDjoh11bmM).

- - -

## Arquitetura

```txt
[Envio do PDF]
     │
     ▼
[PyPDFLoader]
     │  extrai texto por página
     ▼
[Chunkerização: RecursiveCharacterTextSplitter]
     │  chunks (600) + overlap (150)
     ▼
[Embeddings: HuggingFaceEmbeddings]
     │  vetores no espaço semântico
     ▼
[Vector Store: Chroma]
     │  coleção por documento (doc_id)
     ▼
[Retriever: MMR (k=8, fetch_k=40, λ=0.8)]
     │  seleção de passagens diversas e relevantes
     ▼
[Prompt + contexto]
     │  injeta contexto e pergunta no prompt padrão
     ▼
[LLM (Ollama) — qwen2:1.5b]
     │  gera resposta final
     ▼
[UI Streamlit: resposta + trechos + páginas aproximadas]
```

- app.py: UI, estado de sessão, criação da chain e retriever, exibição das fontes.
- index.py: processamento do PDF, chunking, embeddings, criação/abertura do Chroma, salvamento do arquivo

- - -

## Decisões de Design

1. **Chroma local:** simples, rápido, persistente no disco (vector_stores/<doc_id>), ótimo para um projeto reprodutível e com poucos PDFs. 
2. **Embeddings HuggingFace (CPU):** evita dependência de API; **paraphrase-multilingual-MiniLM-L12-v2** é pequeno, multilíngue e eficiente, ideal para execução local e com baixo custo computacional.
3. **MMR no retriever:** reduz redundância entre passagens similares e melhora cobertura de tópicos (evita “eco” dos mesmos trechos).
4. **SLM via Ollama:** qwen2:1.5b oferece baixo custo computacional e bom desempenho para respostas curtas, com fallback simples para modelos maiores (Mistral, Llama 3.x, etc.).
5. **UI/UX:** interface inspirada nos sites oficiais da Ford, o usuário pode acompanhar o progresso do documento e o histórico das conversas, que contém as fonte dos trechos com página aproximada, que ajudam o usuário a encontrar e validar a informação no manual.

- - -

## Engenharia de Prompt

- Fixa o papel de “Assistente Técnico Ford” e restringe a resposta somente ao contexto, com objetivo de impedir alucinações. 
- O fluxo passo a passo classifica a pergunta para o modelo “escolher o modo certo" de responder. 
- Prioriza tabelas e seções técnicas para valores e passos numerados para procedimentos.
- Extração literal e completa para preservar unidades, normas e símbolos, exigindo listar todas as variantes relevantes.
- Em múltiplas correspondências, pede resumo curto + opções, focando em seções específicas quando o item é específico.
- Formato previsível de resposta: começa com resposta direta, depois bullets de variantes/observações e proíbe conteúdo fora do contexto. 

- - -

## Pipeline Detalhada (Indexação + Chunkerização)

1. **Ingestão e salvamento do PDF**
    - O arquivo é salvo em "uploads/<timestamp>-<nome_sanitizado>.pdf".
    - O projeto cria um "doc_id" (hash SHA-1 de 12 caracteres) para identificar a coleção. 

2. **Extração e divisão em chunks**
    - Leitura do PDF com PyPDFLoader.
    - RecursiveCharacterTextSplitter com "chunk_size=600" e "chunk_overlap=150":
        - 600 caracteres por chunk mantém trechos curtos o suficiente para caber no contexto do SLM mesmo com várias passagens.
        - 150 de sobreposição ajuda a não “quebrar” conceitos no meio e melhora recall.

3. **Embeddings e salvamento do arquivo**
    - HuggingFaceEmbeddings roda em CPU por padrão; ideal para desenvolvimento local.
    - Os vetores são armazenados no Chroma, persistidos no diretório vector_stores/<doc_id> para reuso. 

4. **Recuperação + Geração**
    - Retriever MMR com k=8, fetch_k=40, lambda_mult=0.8: bom equilíbrio entre diversidade e top-score.
    - RetrievalQA (LangChain) com chain_type="stuff": os chunks são “colados” diretamente no prompt.
    - Fontes: a UI exibe os trechos usados e a página aproximada para rápida verificação.

- - -

## Embeddings

O modelo de embedding **paraphrase-multilingual-MiniLM-L12-v2** foi escolhido por ser idealmente **multilingue** (inclusive pt-br) e por ser suficientemente **acertivo e rápido** para a aplicação construída: por indexar apenas um único manual (corpus pequeno) e por possuir um prompt construído que exige contexto e respostas estritas.

Abaixo, uma tabela comparativa com outros modelos de embeddings:

| Modelo                                                       | Idioma              | Dim  | Tamanho aprox. | Desempenho prático   | Custo  | Quando usar                                    |
| ------------------------------------------------------------ | ------------------: | ---: | -------------: | -------------------: | -----: | ---------------------------------------------- |
| **paraphrase-multilingual-MiniLM-L12-v2** *(utilizado)*      |    Multi (incl. PT) |  384 |        \~120MB |    Bom/rápido em CPU |    \$0 | **Local leve**, protótipos multilíngues        |
| **intfloat/e5-small-v2**                                     | EN (multi limitado) |  384 |        \~120MB |      Bom em busca EN |    \$0 | CPUs fracas; inglês dominante                  |
| **intfloat/e5-large-v2**                                     |                  EN | 1024 |        \~1.0GB |          Ótimo em EN |    \$0 | Mais qualidade, hardware potente               |
| **BAAI/bge-small-en-v1.5**                                   |                  EN |  384 |        \~140MB |        Bom e estável |    \$0 | Alternativa pequena em EN                      |
| **OpenAI text-embedding-3-small** (API)                      |               Multi | 1536 |              — |            Muito bom |   \$\$ | Quando **latência/qualidade** via API compensa |
| **OpenAI text-embedding-3-large** (API)                      |               Multi | 3072 |              — |            Muito bom | \$\$\$ | Altíssima qualidade, custo maior               |

> “Desempenho prático”: equilíbrio de qualidade e latência para RAG de PDF, variando por hardware/corpus.

- - -

## SLM

O modelo de linguagem **qwen2:1.5b** foi escolhido por ser **leve e rápido** em questão computacional, **rodar localmente e sem necessidade de GPU** e ser **bom em perguntas e respostas objetivas e diretas**.

| Modelo (tag Ollama)            | Parâmetros | Contexto | Recurso típico | Qualidade (RAG curto)                  | Quando usar                          |
| ------------------------------ | ---------: | -------: | -------------- | -------------------------------------- | ------------------------------------ |
| **qwen2:1.5b** *(utilizado)*   |     \~1.5B |     4–8k | CPU ok         | Boa p/ perguntas e respostas objetivas | **Leve e rápido**, dev local         |
| **phi3:3.8b-mini-instruct**    |     \~3.8B |       4k | CPU/GPU leve   | Boa p/ perguntas mais elaboradas       | Para contextos maiores e detalhistas |
| **mistral:7b-instruct**        |       \~7B |       8k | GPU ideal      | Muito boa                              | Quando precisa raciocínio melhor     |
| **llama3.1:8b-instruct**       |       \~8B |    8–16k | GPU ideal      | Muito boa / estável                    | Se tiver GPU e quiser mais qualidade |
| **qwen2.5:7b-instruct**        |       \~7B |      8k+ | GPU ideal      | Muito boa                              | Alternativa moderna e forte          |

- - -

## Como Executar a Aplicação

**Pré-requisitos:** [Python 3.10+](https://www.python.org/downloads/); [Ollama](https://ollama.com/).

1. Clone e entre na pasta:
```bash
git clone <seu-repo>.git
cd <seu-repo>
```

- Essa deve ser a estrutura da pasta:
```pgsql
.
├─ app.py               # UI + chain de QA + retriever MMR + exibição de fontes
├─ index.py             # indexação, chunking, embeddings e salvamento no Chroma
├─ prompt.txt           # prompt padrão do sistema/estilo da resposta
├─ requirements.txt     # arquivo com as dependências da aplicação
├─ style.css            # estilos da UI
├─ uploads/             # PDFs enviados, após o primeiro envio
├─ vector_stores/       # índices Chroma por doc_id, após o primeiro envio
└─ README.md
```

2. Crie "venv" (recomendado):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. Instale dependências do "requirements.txt":
```bash
pip install -r requirements.txt
```

4. Baixe o modelo no Ollama (padrão do projeto):
```bash
ollama pull qwen2:1.5b
```

5. Execute a aplicação:
```bash
streamlit run app.py
```

- - -

## Como usar a aplicação:

1. Envie o PDF do manual do seu veículo (link dos manuais oficiais da Ford para baixar está na tela).
<img src="https://i.imgur.com/I8eekEL.png" alt="Tela de Envio">

2. Aguarde a indexação.
<img src="https://i.imgur.com/48MPlau.png" alt="Tela de Carregamento">

3. Faça perguntas no campo de texto (ex.: “qual óleo devo usar no motor?”).
<img src="https://i.imgur.com/kd92XK6.png" alt="Tela do Chat"> 

4. As perguntas e respostas ficam salvas no campo "Histórico".
<img src="https://i.imgur.com/4tWm5o2.png" alt="Tela do Histórico">

> Você pode verificar de onde a informação foi tirada do manual clicando em "De onde tirei isso".

- - -

## Limitações Conhecidas e Próximos Passos

Limitações:
- Um PDF por vez (coleção por doc_id). Para corpora maiores, inclua múltiplos PDFs na mesma coleção. 
- SLM pequeno pode alucinar se faltarem evidências.

Próximos passos:
- Multi-documento: permitir adicionar vários PDFs à mesma coleção.
- Re-ranking com modelos tipo *bge-reranker*.
- Filtros por metadados (página, seção).
- Avaliação automática com *Ragas* e testes de regressão.

- - -

> Assistente de Manual Ford | Desenvolvido por Cleydson Junior
