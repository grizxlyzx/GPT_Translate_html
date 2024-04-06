def translate_sys_prompt(src_lang, tgt_lang):
    return f'''\
As an expert proficient in language translation.
Your task is to help translate \
{src_lang} text to {tgt_lang} and \
ensuring the translation reflects a high level \
of human expertise.
I'm going to tip $200 for an accurate translation.
'''


def translate_prompt(src_lang, tgt_lang, json_str):
    return f'''\
Translate the following {src_lang} text extracted from a part of a single page, \
from a JSON format into {tgt_lang} whilst keeping function/command \
names(i.e. text in one of Snakecase, Pascalcase or Camelcase and \
any text that doesn\'t form a typical sentence. unchanged:
{json_str}
'''


def restruct_sys_prompt():
    return '''\
As an export proficient in translation, your task is to split translation into parts \
and fit each part into each fields in JSON.
'''


def restruct_prompt(trans_str, ori_str, shreds_str):
    return f'''The translation:
translation: """
{trans_str}
"""
is translated from original text:
text: """
{ori_str}
"""
which is spilt into Segments:
JSON: """
{shreds_str}
"""
Your task is to fit the translation into these segments \
and ensuring no characters in translation is dropped.'''
