import io
import numpy as np
from PIL import Image
import re
# from api.db import LLMType
# from api.db.services.llm_service import LLMBundle
from rag_tokenizer import tokenizer as rag_tokenizer
from vision import OCR

ocr = OCR()

def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    
def chunk(filename, binary=None, tenant_id=None, lang="english", callback=None, **kwargs):
    if binary is None:
        with open(filename, "rb") as f:
            binary = f.read()
            
    img = Image.open(io.BytesIO(binary)).convert('RGB')
    doc = {
        "docnm_kwd": filename,
        "image": img
    }
    bxs = ocr(np.array(img))
    txt = "\n".join([t[0] for _, t in bxs if t[0]])
    eng = lang.lower() == "english"
    callback(0.4, "Finish OCR: (%s ...)" % txt[:12])
    if (eng and len(txt.split(" ")) > 32) or len(txt) > 32:
        tokenize(doc, txt, eng)
        callback(0.8, "OCR results is too long to use CV LLM.")
        return [doc]

    # Note: To implement LLM model here to correct OCR results and describe the picture through CV LLM
    
    # try:
    #     callback(0.4, "Use CV LLM to describe the picture.")
    #     cv_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, lang=lang)
    #     ans = cv_mdl.describe(binary)
    #     callback(0.8, "CV LLM respond: %s ..." % ans[:32])
    #     txt += "\n" + ans
    #     tokenize(doc, txt, eng)
    #     return [doc]
    # except Exception as e:
    #     callback(prog=-1, msg=str(e))

    return []