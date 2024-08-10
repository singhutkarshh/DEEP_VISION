import io
import re
import numpy as np
from rag_tokenizer import tokenizer as rag_tokenizer

# from api.db import LLMType
# from api.db.services.llm_service import LLMBundle


def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])


def chunk(filename, binary, tenant_id, lang, callback=None, **kwargs):
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])

    # is it English
    eng = lang.lower() == "english"  # is_english(sections)
    try:
        callback(0.1, "USE Sequence2Txt LLM to transcription the audio")
        # seq2txt_mdl = LLMBundle(tenant_id, LLMType.SPEECH2TEXT, lang=lang)
        # ans = seq2txt_mdl.transcription(binary)
        # callback(0.8, "Sequence2Txt LLM respond: %s ..." % ans[:32])
        # tokenize(doc, ans, eng)
        # return [doc]
    except Exception as e:
        callback(prog=-1, msg=str(e))

    return []