from bs4 import BeautifulSoup, Tag, NavigableString, Doctype, Comment
from dataclasses import dataclass
from collections import OrderedDict
from openai_api_chat import OpenaiAPIChat
from prompts import *
import json
import asyncio
import re
import math
import difflib
from config import translate_config as conf


@dataclass
class InlineGroup:
    """
    A group of adjacent inline elements extracted from a BeautifulSoup soup object.
    """
    text_shreds: list[NavigableString]
    cids: list[int]
    elements: list[Tag]

    def __post_init__(self):
        assert len(self.text_shreds) == len(self.cids) == len(self.elements)

    def __str__(self):
        return ''.join(self.text_shreds)

    def __len__(self):
        return len(self.text_shreds)


def is_inline_ele(ele: Tag):
    """check if an element is inline or not."""
    return True if ele.name in \
                   {'a', 'abbr', 'acronym', 'b',
                    'bdo', 'big', 'br', 'button',
                    'cite', 'code', 'dfn', 'em', 'i',
                    'img', 'input', 'kbd', 'label',
                    'map', 'object', 'output', 'q',
                    'samp', 'script', 'select', 'small',
                    'span', 'strong', 'sub', 'sup',
                    'textarea', 'time', 'tt', 'var'} else False


def match_score(s1, s2):
    """
    Calculates the similarity between two strings.
    Returns 1.0 if two strings matches, ignore casing,
    symbols and spacing.
    :param s1: string 1 for comparison
    :param s2: string 2 for comparison
    :return: match score, 1.0 for best match.
    """
    compact_tab = str.maketrans({
        ' ': '',
        '\n': '',
        '\t': '',
        ',': '',
        '，': '',
        '.': ''
    })
    s1 = s1.translate(compact_tab).lower()
    s2 = s2.translate(compact_tab).lower()
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def get_text_group_inline(
        element: Tag
) -> OrderedDict[str, InlineGroup]:
    """
    Extracts and groups adjacent inline elements from a BeautifulSoup soup object,
    based on the inductive bias that adjacent inline elements tend to form complete sentences.
    :param element: BeautifulSoup Tag object representing the root element to extract inline elements from
    :return: OrderedDict containing grouped inline elements
    """

    all_groups = []
    cnt = 0
    ignore_char_set = {
        '\n', '\xa0'
    }

    def traversal(ele: Tag):
        nonlocal cnt, all_groups
        group = []
        inline = is_inline_ele(ele)
        for cid, c in enumerate(ele.contents):
            if isinstance(c, (Doctype, Comment)) or c.name == 'script':
                continue
            elif isinstance(c, NavigableString) and not set(c).issubset(ignore_char_set):
                group.append([c.replace('\n', ''), cid, ele, cnt])
                cnt += 1
            elif isinstance(c, Tag):
                sub_shreds = traversal(c)
                if sub_shreds is not None:  # non-None return indicates inline child tag
                    group += sub_shreds
                elif (not inline) and group:
                    all_groups.append(group)
                    group = []
        if not inline:
            if group:
                all_groups.append(group)
            return None
        else:
            return group

    traversal(element)
    all_groups.sort(key=lambda x: x[0][3])
    groups_map = OrderedDict({})
    for i, g in enumerate(all_groups):
        groups_map[str(i)] = InlineGroup(
            [item[0] for item in g],
            [item[1] for item in g],
            [item[2] for item in g]
        )
    return groups_map


def as_json_obj(
        raw: str
):
    regex = r'\{.*\}'
    matches = re.findall(regex, raw, re.DOTALL)
    j_str = matches[0] if matches else ''
    j_str = re.sub(r',\s*}', '}', j_str)
    ret = None
    try:
        ret = json.loads(j_str)
    except Exception:
        pass
    return ret


def validate_fit_in(
        shreds_in: dict[str, str],
        trans_str: str,
        shreds_out: dict[str, str],
) -> (float, str):
    """
    Validates if translated text fit correctly into the original structure.
    :param shreds_in: dict of pieces of original text before translation
    :param trans_str: translated text of grouped inline shreds
    :param shreds_out: dict of pieces of translated text
    :return: A tuple of a fit score and the reason of not perfectly fit.
             score of 1 indicates perfectly fit, 0 indicates it is not
             able to fit at all, score between 0 and 1 indicate partially fit.
    """
    if len(shreds_in) != len(shreds_out):
        return 0., f'Length not match, in({len(shreds_in)}) != out({len(shreds_out)}).'

    sorted_shreds_out = {k: v for k, v in sorted(shreds_out.items(), key=lambda x: int(x[0]))}
    fit_str = ''.join([v for v in sorted_shreds_out.values()])
    if (score := match_score(trans_str, fit_str)) != 1.0:
        return score, f'String not match, to_fit="{trans_str}" | fit="{fit_str}"'
    return 1., ''


async def restruct(
        group: InlineGroup,
        ori: str,
        trans: str
):
    """
    Restructures the translated text to fit the original structure.
    :param group: inline group to be fit back into
    :param ori: original grouped text before translation
    :param trans: translated text
    :return: restruct result
    """
    max_retry = 10
    retry = 0
    chat = OpenaiAPIChat(
        model_name=conf.RESTRUCT_MODEL,
        system_prompt=restruct_sys_prompt()
    )
    shreds_in = OrderedDict({})
    for i, shred in enumerate(group.text_shreds):
        shreds_in[str(i)] = shred
    shreds_in_str = json.dumps(shreds_in, ensure_ascii=False, indent=0)
    p = restruct_prompt(trans, ori, shreds_in_str)
    temperature = 0.01
    fit_candidates = []
    while True:
        try:
            chat.clear()
            response = ''
            async for chunk, stop_reason in chat.get_stream_aresponse(p, json_response=True, temperature=temperature):
                response += chunk
            shreds_out = as_json_obj(response)
            # response validation check
            if not shreds_out:
                raise ValueError('Invalid model response as JSON object.')
            score, err = validate_fit_in(
                shreds_in,
                trans,
                shreds_out
            )
            fit_candidates.append([score, shreds_out])
            if err:
                raise ValueError(err)
            break

        except Exception as e:
            retry += 1
            temperature *= 1.6  # exponential increase temperature
            if retry > max_retry:
                break

    # replace contents
    if fit_candidates:
        max_score, shreds_out = max(fit_candidates, key=lambda x: x[0])
        for k, v in shreds_out.items():
            cid = group.cids[int(k)]
            ele = group.elements[int(k)]
            ele.contents[cid].replace_with(v)
        return 'C' if max_score < 1.0 else 'S'
    else:
        return 'F'


async def group_fit_in(
        group: InlineGroup,
        ori: str,
        trans: str
):
    """
    Fits translated text into the original structure.
    :param group: inline group to be fit back into
    :param ori: original grouped text before translation
    :param trans: translated text
    :return: fit-in result
    """
    if match_score(str(group), trans) == 1.0:  # no translation is needed, e.g. function name, special symbols, etc.
        return 'S'
    elif len(group) == 1:  # single element text
        str_content = group.elements[0].contents[group.cids[0]]
        str_content.replace_with(trans)
        return 'S'
    else:  # multi element text: restruct is needed
        return await restruct(group, ori, trans)


def segment_groups_map(
        groups_map: dict[str, InlineGroup],
        max_token: int,
        token_counter: callable
) -> list[OrderedDict]:
    """
    Segments a map of inline groups based on token count, hoping the
    translation response of each segment won't exceed context length
    of the model. Also, run each segments currently also speeds up the job.
    :param groups_map: Dictionary of inline groups and their id
    :param max_token: max number of token in each segment
    :param token_counter: a function that takes a string as input and output number of tokens
    :return: A list of segmented inline groups
    """
    if not (token_all := sum([token_counter(str(g)) for g in groups_map.values()])):
        return []
    n_seg = math.ceil(token_all / max_token)
    len_seg = math.ceil(token_all / n_seg)
    ret = []
    token_cnt = 0
    cnt = 0
    seg = OrderedDict({})
    for k, group in groups_map.items():
        n = token_counter(str(group))
        if n > max_token:
            # raise ValueError(f'Length of single paragraph [{n}] exceed max length [{max_token}].')
            print(f'Single paragraph exceed max length [{n} > {max_token}]. Skip this one!')
            continue
        if (token_cnt > len_seg) and seg:
            ret.append(seg)
            token_cnt = 0
            cnt = 0
            seg = OrderedDict({})
        seg[str(cnt)] = group
        cnt += 1
        token_cnt += n
    ret.append(seg)
    return ret


async def translate_groups(
        groups_map: OrderedDict[str, InlineGroup],
):
    """
    Performs translation and restruct task on one segment of all inline groups.
    :param groups_map: OrderedDict of inline groups and corresponding id
    :return: list of results of each task,
             "S" for success;
             "C" for Compromise, which means cannot perfectly restruct the paragraph;
             "F" for Fail;
             or other exceptions
    """
    groups_in = {
        k: str(v).replace('\n', '') for k, v in groups_map.items()
    }
    groups_in_str = json.dumps(groups_in, indent=0, ensure_ascii=False)

    chat = OpenaiAPIChat(
        model_name=conf.TRANSLATE_MODEL,
        system_prompt=translate_sys_prompt(conf.SOURCE_LANGUAGE, conf.TARGET_LANGUAGE)
    )
    response, stop_reason = '', ''
    try:
        p = translate_prompt(conf.SOURCE_LANGUAGE, conf.TARGET_LANGUAGE, groups_in_str)
        async for chunk, stop_reason in chat.get_stream_aresponse(p, json_response=True, temperature=0.01):
            response += chunk
        if stop_reason == 'length':
            raise RuntimeError
    except RuntimeError:
        pass  # TODO: maybe redo segmentation with less tokens and try again
    groups_out = as_json_obj(response)
    fit_in_tasks = []
    for i, trans in groups_out.items():
        group = groups_map[i]
        fit_in_tasks.append(group_fit_in(group, groups_in[i], trans))
    results = await asyncio.gather(*fit_in_tasks, return_exceptions=True)
    return results


async def translation_pipeline(
        soup: BeautifulSoup
) -> list[str]:
    """
    Main entry point to translates the HTML in the soup object in place.
    The pipeline is composed of 3 steps:
    1. Traverse the DOM, groups up adjacent inline elements into InlineGroup,
        since, conventionally, these elements are more likely to form a semantically
        complete sentence or paragraph, thus makes the model more likely to generate
        a better translation.
    2. Split the inline groups to be translated into smaller translation tasks, so that
        the response won't exceed model's context length, then, run each tasks concurrently.
    3. After a translation tasks are done, use the model to split each translated
        inline group and fit each piece back to inline tags to recover the structure
        of the DOM.

    :param soup: The BeautifulSoup object representing the HTML to be translated
    :return: list of results of translation segments,
             "S" for success;
             "C" for Compromise, which means cannot perfectly restruct the paragraph;
             "F" for Fail, or any other exceptions
    """
    groups_map = get_text_group_inline(soup)
    groups_map_segments = segment_groups_map(
        groups_map,
        int(conf.N_INPUT_TOKEN),
        OpenaiAPIChat(conf.TRANSLATE_MODEL).n_tokens
    )
    tasks = [translate_groups(seg) for seg in groups_map_segments]
    results = await asyncio.gather(*tasks)
    results = [j for i in results for j in i]
    return results


if __name__ == '__main__':
    # example usage
    p_in = './example/installation.html'
    p_out = './example/out.html'
    with open(p_in, 'r') as fin:
        html_in = fin.read()
    bs = BeautifulSoup(html_in, 'html.parser')
    ret = asyncio.run(translation_pipeline(bs))
    with open(p_out, 'w') as fout:
        fout.write(str(bs))
