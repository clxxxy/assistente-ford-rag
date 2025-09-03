# Assistente de Manual Ford — Pipeline RAG
Uma aplicação com interface Streamlit que permite carregar um PDF, indexá-lo em um Vector Store Chroma com embeddings HuggingFace, e consultar via SLM local (Ollama) com recuperação (RAG).

## Sumário

1. [Visão Geral](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#vis%C3%A3o-geral)
2. [Arquitetura](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#arquitetura)
3. [Decisões de Design](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#decis%C3%B5es-de-design)
4. [Engenharia de Prompt](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#engenharia-de-prompt)
5. [Pipeline Detalhada](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#pipeline-detalhada-indexa%C3%A7%C3%A3o--chunkeriza%C3%A7%C3%A3o)

## Visão geral

Este projeto demonstra um RAG para perguntas e respostas: o usuário o manual do veículo do veículo Ford em PDF, o sistema fragmenta o conteúdo (chunks), gera embeddings e salva em um ChromaDB, uma cadeia de recuperação consulta os trechos mais relevantes e monta um prompt para o SLM responder com as fontes.

- Frontend: Streamlit com UI para upload, progresso de indexação e histórico de Q&A. 
- Indexação: PyPDFLoader → RecursiveCharacterTextSplitter → HuggingFaceEmbeddings → Chroma. 
- Recuperação: Retriever com MMR para equilíbrio entre relevância e diversidade.
- SLM local: qwen2:1.5b via Ollama (local).

## Arquitetura

- app.py: UI, estado de sessão, criação da chain e retriever, exibição das fontes.
- index.py: processamento do PDF, chunking, embeddings, criação/abertura do Chroma, salvamento do arquivo

## Decisões de Design

1. **Chroma local:** simples, rápido, persistente no disco (vector_stores/<doc_id>), ótimo para um projeto reprodutível e com poucos PDFs. 
2. **Embeddings HuggingFace (CPU):** evita dependência de API; **paraphrase-multilingual-MiniLM-L12-v2** é pequeno, multilíngue e eficiente, ideal para execução local e com baixo custo computacional.
3. **MMR no retriever:** reduz redundância entre passagens similares e melhora cobertura de tópicos (evita “eco” dos mesmos trechos).
4. **SLM via Ollama:** qwen2:1.5b oferece baixo custo computacional e bom desempenho para respostas curtas, com fallback simples para modelos maiores (Mistral, Llama 3.x, etc.).
5. **UI/UX:** interface inspirada nos sites oficiais da Ford, o usuário pode acompanhar o progresso do documento e o histórico das conversas, que contém as fonte dos trechos com página aproximada, que ajudam o usuário a encontrar e validar a informação no manual.

## Engenharia de Prompt

- Fixa o papel de “Assistente Técnico Ford” e restringe a resposta somente ao contexto, com objetivo de impedir alucinações. 
- O fluxo passo a passo classifica a pergunta para o modelo “escolher o modo certo" de responder. 
- Prioriza tabelas e seções técnicas para valores e passos numerados para procedimentos.
- Extração literal e completa para preservar unidades, normas e símbolos, exigindo listar todas as variantes relevantes.
- Em múltiplas correspondências, pede resumo curto + opções, focando em seções específicas quando o item é específico.
- Formato previsível de resposta: começa com resposta direta, depois bullets de variantes/observações e proíbe conteúdo fora do contexto. 

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

## Embeddings e SLM

O modelo de embedding **paraphrase-multilingual-MiniLM-L12-v2** foi escolhido seguindo os seguintes parâmetros: Idioma, dimensionalidade, tamanho, licença, desempenho prático, custo e uso.

> Os números de desempenho são gerais (MTEB/uso prático) e variam por hardware e corpus.

| Modelo (HF/API)                                           |              Idioma |  Dim | Tamanho aprox. | Licença      | Desempenho prático |  Custo | Quando usar                                    |
| --------------------------------------------------------- | ------------------: | ---: | -------------: | ------------ | -------------------: | -----: | ---------------------------------------------- |
| **paraphrase-multilingual-MiniLM-L12-v2** (HF) *(utilizado)* |    Multi (incl. PT) |  384 |        \~120MB | permissiva   |    Bom/rápido em CPU |    \$0 | **Local leve**, protótipos multilíngues        |
| **intfloat/e5-small-v2** (HF)                             | EN (multi limitado) |  384 |        \~120MB | Apache-2.0   |      Bom em busca EN |    \$0 | CPUs fracas; inglês dominante                  |
| **intfloat/e5-large-v2** (HF)                             |                  EN | 1024 |        \~1.0GB | Apache-2.0   |          Ótimo em EN |    \$0 | Mais qualidade, máquina potente                |
| **BAAI/bge-small-en-v1.5** (HF)                           |                  EN |  384 |        \~140MB | MIT          |        Bom e estável |    \$0 | Alternativa small em EN                        |
| **BAAI/bge-m3** (HF)                                      |       Multi (forte) | 1024 |        \~1.3GB | MIT          |      Excelente multi |    \$0 | **Alta qualidade**, precisa de RAM             |
| **OpenAI text-embedding-3-small** (API)                   |               Multi | 1536 |              — | Proprietária |            Muito bom |   \$\$ | Quando **latência/qualidade** via API compensa |
| **OpenAI text-embedding-3-large** (API)                   |               Multi | 3072 |              — | Proprietária |                 SOTA | \$\$\$ | Altíssima qualidade, custo maior               |
