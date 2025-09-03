# Assistente de Manual Ford — Pipeline RAG
Uma aplicação com interface Streamlit que permite carregar um PDF, indexá-lo em um Vector Store Chroma com embeddings HuggingFace, e consultar via SLM local (Ollama) com recuperação (RAG).

## Sumário

1. [Visão Geral](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#vis%C3%A3o-geral)
2. [Arquitetura](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#arquitetura)
3. [Decisões de Design](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#decis%C3%B5es-de-design)
4. [Engenharia de Prompt](https://github.com/clxxxy/assistente-ford-rag/edit/main/README.md#engenharia-de-prompt)

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
