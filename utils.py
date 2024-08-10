import tiktoken
import copy
import re

from PIL import Image
from rag_tokenizer import tokenizer as rag_tokenizer

all_codecs = [
    'utf-8', 'gb2312', 'gbk', 'utf_16', 'ascii', 'big5', 'big5hkscs',
    'cp037', 'cp273', 'cp424', 'cp437',
    'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857',
    'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869',
    'cp874', 'cp875', 'cp932', 'cp949', 'cp950', 'cp1006', 'cp1026', 'cp1125',
    'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256',
    'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr',
    'gb2312', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
    'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1',
    'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7',
    'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_13',
    'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t', 'koi8_u',
    'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman',
    'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213',
    'utf_32', 'utf_32_be', 'utf_32_le''utf_16_be', 'utf_16_le', 'utf_7'
]


def find_codec(blob):
    global all_codecs
    for c in all_codecs:
        try:
            blob[:1024].decode(c)
            return c
        except Exception as e:
            pass
        try:
            blob.decode(c)
            return c
        except Exception as e:
            pass
    return "utf-8"

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoder =  tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoder.encode(string))
        return num_tokens
    except Exception as e:
        pass
    return 0

def concat_img(img1, img2):
    if img1 and not img2:
        return img1
    if not img1 and img2:
        return img2
    if not img1 and not img2:
        return None
    width1, height1 = img1.size
    width2, height2 = img2.size

    new_width = max(width1, width2)
    new_height = height1 + height2
    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (0, height1))

    return new_image

def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    
def add_positions(d, poss):
    if not poss:
        return
    d["page_num_int"] = []
    d["position_int"] = []
    d["top_int"] = []
    for pn, left, right, top, bottom in poss:
        d["page_num_int"].append(int(pn + 1))
        d["top_int"].append(int(top))
        d["position_int"].append((int(pn + 1), int(left), int(right), int(top), int(bottom)))


def tokenize_table(tbls, doc, eng, batch_size=10):
    res = []
    # add tables
    for (img, rows), poss in tbls:
        if not rows:
            continue
        if isinstance(rows, str):
            d = copy.deepcopy(doc)
            tokenize(d, rows, eng)
            d["content_with_weight"] = rows
            if img: d["image"] = img
            if poss: add_positions(d, poss)
            res.append(d)
            continue
        de = "; " if eng else "； "
        for i in range(0, len(rows), batch_size):
            d = copy.deepcopy(doc)
            r = de.join(rows[i:i + batch_size])
            tokenize(d, r, eng)
            d["image"] = img
            add_positions(d, poss)
            res.append(d)
    return res

def naive_merge_docx(sections, chunk_token_num=128, delimiter="\n。；！？"):
    if not sections:
        return [], []

    cks = [""]
    images = [None]
    tk_nums = [0]

    def add_chunk(t, image, pos=""):
        nonlocal cks, tk_nums, delimiter
        tnum = num_tokens_from_string(t)
        if tnum < 8:
            pos = ""
        if tk_nums[-1] > chunk_token_num:
            if t.find(pos) < 0:
                t += pos
            cks.append(t)
            images.append(image)
            tk_nums.append(tnum)
        else:
            if cks[-1].find(pos) < 0:
                t += pos
            cks[-1] += t
            images[-1] = concat_img(images[-1], image)
            tk_nums[-1] += tnum

    for sec, image in sections:
        add_chunk(sec, image, '')

    return cks, images

def tokenize_chunks_docx(chunks, doc, eng, images):
    res = []
    # wrap up as es documents
    for ck, image in zip(chunks, images):
        if len(ck.strip()) == 0:continue
        print("--", ck)
        d = copy.deepcopy(doc)
        d["image"] = image
        tokenize(d, ck, eng)
        res.append(d)
    return res
