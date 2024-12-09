import torch
import pandas as pd
import json
import sys

from rank_bm25 import BM25Okapi, BM25Plus

from transformers import T5ForConditionalGeneration

from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5



def retrieving(query, bm25, corpus, top_n=3):
    tokenized_query = query.split(" ")
    # doc_scores = bm25.get_scores(tokenized_query)
    return bm25.get_top_n(tokenized_query, corpus, n=top_n)


def run_retrieval(src_dir, claim_file, bm25_model):

    claims = pd.read_json(src_dir+claim_file, lines=True)
    claims = list(claims['claim'])

    print("--> claim_file:", claim_file)

    ret_data = []

    cnt = 0
    for c in claims:
        y= [' '.join(z) for z in retrieving(c, bm25_model, corpus, top_n=100)]
        ret_data.append({'claim': c, 'retrieved_doc': y})
        cnt += 1
        if cnt % 30 == 0:
            print(cnt)

    json.dump(ret_data, open(src_dir+'bm25Plus-'+fname+'-top100.json', 'w'), indent="\t")


def reranking(src_dir, fname, reranker):
    results = []
    print("--> reranking data in", fname)
    with open(src_dir+fname) as f:
        data = json.load(f)
        count = 0
        for item in data:
            if count % 5 == 0:
                print(count, flush=True)
            count += 1

            x = {}
            x['claim'] = item['claim']
            query = Query(item['claim'])
            # texts = [Text(d) for d in item['retrieved_doc']]
            texts = [Text(p, {'docid': "", "title": "", "text": p}, 0) for p in item['retrieved_doc']]
            reranked = reranker.rerank(query, texts)
            reranked.sort(key=lambda x: x.score, reverse=True)
            x['documents'] = [text.metadata for text in reranked]
            # x['documents'] = reranked[0].metadata
            results.append(x)
            if count % 1000 == 0:
                json.dump(results, open(src_dir+"rerank_"+str(count)+"_"+fname,'w'))

    print("--> Writing to", src_dir+"rerank_"+fname)
    json.dump(results, open(src+"rerank_"+fname,'w'))


if __name__ == "__main__":
    # corpus_file = "./data/corpus.jsonl"
    src = './data/'
    corpus_file = src+sys.argv[1].lower()

    print("--> corpus_file:", corpus_file)

    df = pd.read_json(corpus_file, lines=True)
    # print(len(list(df['text'])))

    corpus = list(df['text'])
    #corpus = corpus[20:320] # for testing

    tokenized_corpus = [(' '.join(doc)).split(" ") for doc in corpus]

    # bm25 = BM25Okapi(tokenized_corpus)
    bm25 = BM25Plus(tokenized_corpus)

    fname = sys.argv[2].lower()
    claim_file = fname+".jsonl"

    run_retrieval(src, claim_file, bm25)

    reranker_model = T5ForConditionalGeneration.from_pretrained("castorini/monot5-3b-msmarco-10k", torch_dtype=torch.float16)

    device = torch.device("cuda")
    reranker_model.to(device)

    # print("MODEL LOADED!", flush=True)


    reranker = MonoT5(model=reranker_model)

    fname = 'bm25Plus-'+fname+'-top100.json'

    reranking(src, fname, reranker)
