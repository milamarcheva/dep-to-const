import os
from pathlib import Path
import pyconll
from nltk.tree import *
import unicodedata

class NonProjError(Exception):
    pass


class RootNonProjError(Exception):
    pass


class CFContainedError(Exception):
    pass


class ContainNoneError(Exception):
    pass


class NotContainRootError(Exception):
    pass


def keep_token(token, exclude_punct=False):
    if token.is_multiword():
        return False
    if not exclude_punct:
        return True
    return not (token.upos == 'PUNCT' and token.form != '-')


def sentence_tokens(sentence, exclude_punct=False):
    return [token for token in sentence if keep_token(token, exclude_punct)]


def nonprojective_included(sentence):
    kept_ids = {int(token.id) for token in sentence}
    arcs = []
    for token in sentence:
        try:
            dep = int(token.id)
            head = int(token.head)
        except (TypeError, ValueError):
            continue
        if head == 0:
            continue
        # If the head was filtered out (e.g., punctuation removal), ignore the arc.
        if head not in kept_ids:
            continue
        start, end = sorted((dep, head))
        arcs.append((start, end))
    for i in range(len(arcs)):
        a_start, a_end = arcs[i]
        for j in range(i + 1, len(arcs)):
            b_start, b_end = arcs[j]
            if (a_start < b_start < a_end < b_end) or (b_start < a_start < b_end < a_end):
                return True
    return False

# Since pyconll package sometimes fail to censor non-projective dependency tree that contains crossing above root edge,
# function for handling this exception is defined here
def rootcross_included(sentence):
    if extract_head(sentence) is None:
        raise NotContainRootError
    root_id = int(extract_head(sentence).id)

    def is_crossing_root(token_id, head_id):
        return (token_id < root_id
                and root_id < head_id) or (token_id > root_id
                                           and root_id > head_id)

    for token in sentence:
        if token.is_multiword():
            continue
        try:
            head_id = int(token.head)
        except (TypeError, ValueError):
            continue
        if is_crossing_root(int(token.id), head_id):
            return True

    return False


def Cf_included(s):
    if s is None:
        raise ContainNoneError
    for c in s:
        if unicodedata.category(c) == "Cf":
            return True
    return False


def sentence_to_str(sentence):
    s = ""
    for token in sentence:
        if token.form is not None:
            s += token.form
            s += " "
    return s.rstrip()


def get_token_with_id(sentence, token_id):
    for token in sentence:
        if str(token_id) == token.id:
            return token


def extract_head(sentence):
    for token in sentence:
        if token.deprel == 'root':
            return token


def extract_children(sentence, parent_token):
    child_list = [int(parent_token.id)]
    for token in sentence:
        if token.head == parent_token.id:
            child_list.append(int(token.id))
    return sorted(child_list)


def extract_left_children(sentence, parent_token):
    child_list = extract_children(sentence, parent_token)
    return list(filter(lambda c_id: c_id < int(parent_token.id), child_list))


def extract_right_children(sentence, parent_token):
    child_list = extract_children(sentence, parent_token)
    return list(filter(lambda c_id: c_id > int(parent_token.id), child_list))


# 1. If a sentence includes control character, then raise Error because preprocess.py does not expect it. (This is not a sanitization, so ideally this procedure should be in other function.)
# 2. Convert all the parens into -LRB- or -RRB- to resolve ambiguity of phrase structure.
# 3. Remove space in form because preprocess.py expect each forms do not contain any space.
# - Space is contained at least in French_GSD, but they are numeric, therefore possibly not problematic.
def sanitize_form(form):
    if Cf_included(form):
        raise CFContainedError
    # if ' ' in form:
    #     logger.info(f'Space included in the form: {form}')
    return form.replace('(', '-LRB-').replace(')', '-RRB-').replace('（', '-LRB-').replace('）', '-RRB-').replace(' ', '')


def create_leaf(preterminal_nt, form):
    return f'({preterminal_nt} {sanitize_form(form)}) '


def create_leaf_with_Tree(preterminal_nt, form):
    return Tree(preterminal_nt, [sanitize_form(form)])


def get_X_nt(token, preterminal=False):
    return 'X'


def get_pos_nt(token, preterminal=False):
    if preterminal:
        return f'{token.upos}'
    return f'{token.upos}P'


def get_merge_pos_nt(token, preterminal=False):
    return get_pos_nt(token, preterminal).replace('PRON', 'NOUN').replace(
        'PROPN', 'NOUN').replace('DET', 'NOUN')


def get_dep_nt(token, preterminal=False):
    return f'{token.deprel}'


def flat_converter(sentence, token, get_nt):
    children = extract_children(sentence, token)
    if len(children) == 1:
        return create_leaf(get_nt(token, preterminal=True), token.form)
    constituency = f'({get_nt(token)} '
    for child_id in children:
        if child_id == int(token.id):
            sub_constituency = create_leaf(get_nt(token, preterminal=True), token.form)
        else:
            sub_constituency = flat_converter(
                sentence, get_token_with_id(sentence, child_id), get_nt)
        constituency += sub_constituency
    return constituency.rstrip() + ') '


def make_phrase_from_left(sentence, token, left_children_ids,
                          right_children_ids, get_nt):
    if left_children_ids == []:
        if right_children_ids == []:
            return create_leaf_with_Tree(get_nt(token, preterminal=True), token.form)
        else:
            r_token = get_token_with_id(sentence, right_children_ids.pop(-1))
            return Tree(get_nt(token), [
                make_phrase_from_left(sentence, token, left_children_ids,
                                      right_children_ids, get_nt),
                make_phrase_from_left(
                    sentence, r_token, extract_left_children(
                        sentence, r_token),
                    extract_right_children(sentence, r_token), get_nt)
            ])

    l_token = get_token_with_id(sentence, left_children_ids.pop(0))
    return Tree(get_nt(token), [
        make_phrase_from_left(
            sentence, l_token, extract_left_children(sentence, l_token),
            extract_right_children(sentence, l_token), get_nt),
        make_phrase_from_left(sentence, token, left_children_ids,
                              right_children_ids, get_nt)
    ])


def make_phrase_from_right(sentence, token, left_children_ids,
                           right_children_ids, get_nt):
    if right_children_ids == []:
        if left_children_ids == []:
            return create_leaf_with_Tree(get_nt(token, preterminal=True), token.form)
        else:
            l_token = get_token_with_id(sentence, left_children_ids.pop(0))
            return Tree(get_nt(token), [
                make_phrase_from_right(
                    sentence, l_token, extract_left_children(
                        sentence, l_token),
                    extract_right_children(sentence, l_token), get_nt),
                make_phrase_from_right(sentence, token, left_children_ids,
                                       right_children_ids, get_nt),
            ])
    r_token = get_token_with_id(sentence, right_children_ids.pop(-1))
    return Tree(get_nt(token), [
        make_phrase_from_right(sentence, token, left_children_ids,
                               right_children_ids, get_nt),
        make_phrase_from_right(
            sentence, r_token, extract_left_children(sentence, r_token),
            extract_right_children(sentence, r_token), get_nt)
    ])


def left_converter(sentence, head_token, get_nt):
    return make_phrase_from_left(sentence, head_token,
                                 extract_left_children(sentence, head_token),
                                 extract_right_children(sentence, head_token),
                                 get_nt).pformat(margin=1e100)


def right_converter(sentence, head_token, get_nt):
    return make_phrase_from_right(sentence, head_token,
                                  extract_left_children(sentence, head_token),
                                  extract_right_children(sentence, head_token),
                                  get_nt).pformat(margin=1e100)


def general_converter(converter, sentence, get_nt, exclude_punct=False):
    filtered_sentence = sentence_tokens(sentence, exclude_punct)
    if len(filtered_sentence) == 0:
        raise NotContainRootError
    if nonprojective_included(filtered_sentence):
        raise NonProjError
    if rootcross_included(filtered_sentence):
        raise RootNonProjError
    head_token = extract_head(filtered_sentence)
    if head_token is None:
        raise NotContainRootError
    return converter(filtered_sentence, head_token, get_nt).rstrip()


def generate_tokens(sentence, exclude_punct=False):
    plain_sentence = ""
    for token in sentence:
        if not keep_token(token, exclude_punct):
            continue
        plain_sentence += sanitize_form(token.form) + ' '
    return plain_sentence.rstrip()


def setup_functions(args):
    if args.convert_method == 'flat':
        converter = flat_converter
    elif args.convert_method == 'left':
        converter = left_converter
    elif args.convert_method == 'right':
        converter = right_converter

    if args.without_label:
        get_nt = get_X_nt
    elif args.use_pos_label:
        get_nt = get_pos_nt
    elif args.use_merged_pos_label:
        get_nt = get_merge_pos_nt
    elif args.use_dep_label:
        get_nt = get_dep_nt

    return converter, get_nt


def get_method_str(args):
    convert_method_str = args.convert_method
    if args.without_label:
        label_method_str = 'X'
    elif args.use_pos_label:
        label_method_str = 'POS'
    elif args.use_merged_pos_label:
        label_method_str = 'M_POS'
    elif args.use_dep_label:
        label_method_str = 'DEP'
    return f'{convert_method_str}-{label_method_str}'


def find_conllu_files(source_path):
    p = Path(source_path)
    if p.is_file() and p.suffix == '.conllu':
        return [p]
    return [conllu_file for conllu_file in p.glob('**/*.conllu')]


def generate_path_info(args):
    files_to_convert = find_conllu_files(args.source_path)
    method_str = get_method_str(args)
    return files_to_convert, method_str, os.path.join(args.output_path,
                                                      method_str)
