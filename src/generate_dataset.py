import argparse
import os
import tempfile
from converter import *
import pyconll

from logging import getLogger, FileHandler, Formatter, DEBUG
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.propagate = False

parser = argparse.ArgumentParser()

# directory parameters
parser.add_argument('--source_path',
                    default='../../../resource/ud-treebanks-v2.7/English_EWT')
parser.add_argument('--G18_conllid_file',
                    default='')
parser.add_argument('--output_path',
                    default='../../../resource/ud-converted')

# method specification parameters
parser.add_argument('--convert_method', default='flat')
parser.add_argument('--without_label', action='store_true')
parser.add_argument('--use_pos_label', action='store_true')
parser.add_argument('--use_merged_pos_label', action='store_true')
parser.add_argument('--use_dep_label', action='store_true')

# other parameter(s)
parser.add_argument('--dev_test_sentence_num', default=5000, type=int)
parser.add_argument('--train_token_num', default=40000000, type=int)
parser.add_argument('--write_deptree', action='store_true')
parser.add_argument('--exclude_punct', action='store_true',
                    help='Exclude punctuation from parse and token outputs, except "-"')
parser.add_argument('--add_root', action='store_true',
                    help='Wrap each generated parse with a top-level ROOT node.')


def convert_conllu_files(args):
    conllu_files_to_convert, method_str, output_dir = generate_path_info(args)

    converter, get_nt = setup_functions(args)
    if args.write_deptree:
        original_deptree_dir = Path(os.path.join(args.output_path, "original_deptree"))
        if not original_deptree_dir.exists():
            original_deptree_dir.mkdir()

    for conllu_file in conllu_files_to_convert:
        logger.info(f'Converting {conllu_file.name} with {method_str} method.')
        corpus = pyconll.load_from_file(str(conllu_file))
        
        processed_sentence_num = 0
        token_num = 0

        inclempty_count = 0
        cfcontained_count = 0
        contain_none_count = 0
        nonproj_count = 0
        root_nonproj_count = 0
        notcontainroot_count = 0

        if args.write_deptree:
            with open(os.path.join(output_dir, f'{conllu_file.stem}.txt'), 'w') as f, \
                 open(os.path.join(output_dir,f'{conllu_file.stem}.tokens'), 'w') as g, \
                 original_deptree_dir.joinpath(conllu_file.name).open('w') as h:
                for i, sentence in enumerate(corpus):
                    if i % 100000 == 0:
                        logger.info(f'{i} data has been converted.')
                    try:
                        phrase_structure = general_converter(
                            converter, sentence, get_nt, args.exclude_punct)
                        kept_tokens = sentence_tokens(sentence, args.exclude_punct)
                        if len(kept_tokens) == 1:
                            phrase_structure = f'({get_nt(kept_tokens[0])} {phrase_structure})'
                        if args.add_root:
                            phrase_structure = f'(ROOT {phrase_structure})'
                        f.write(phrase_structure)
                        f.write('\n')
                        g.write(generate_tokens(sentence, args.exclude_punct))
                        g.write('\n')
                        h.write(sentence.conll())
                        h.write('\n\n')

                        processed_sentence_num += 1
                        token_num += len(sentence)
                        if "train" not in conllu_file.stem and processed_sentence_num == args.dev_test_sentence_num:
                            break
                        if "train" in conllu_file.stem and token_num > args.train_token_num:
                            break
                    except KeyError:
                        inclempty_count += 1
                        continue
                    except ContainNoneError:
                        contain_none_count += 1
                        continue
                    except NonProjError:
                        nonproj_count += 1
                        continue
                    except RootNonProjError:
                        root_nonproj_count += 1
                        continue
                    except CFContainedError:
                        logger.info(f'Cf contained in {sentence_to_str(sentence)}')
                        cfcontained_count += 1
                        continue
                    except NotContainRootError:
                        notcontainroot_count += 1
                        continue
        else:
            with open(os.path.join(output_dir, f'{conllu_file.stem}.txt'), 'w') as f, \
                 open(os.path.join(output_dir,f'{conllu_file.stem}.tokens'), 'w') as g:
                for i, sentence in enumerate(corpus):
                    if i % 100000 == 0:
                        logger.info(f'{i} data has been converted.')
                    try:
                        phrase_structure = general_converter(
                            converter, sentence, get_nt, args.exclude_punct)
                        kept_tokens = sentence_tokens(sentence, args.exclude_punct)
                        if len(kept_tokens) == 1:
                            phrase_structure = f'({get_nt(kept_tokens[0])} {phrase_structure})'
                        if args.add_root:
                            phrase_structure = f'(ROOT {phrase_structure})'
                        f.write(phrase_structure)
                        f.write('\n')
                        g.write(generate_tokens(sentence, args.exclude_punct))
                        g.write('\n')

                        processed_sentence_num += 1
                        token_num += len(sentence)
                        if "train" not in conllu_file.stem and processed_sentence_num == args.dev_test_sentence_num:
                            break
                        if "train" in conllu_file.stem and token_num > args.train_token_num:
                            break
                    except KeyError:
                        inclempty_count += 1
                        continue
                    except ContainNoneError:
                        contain_none_count += 1
                        continue
                    except NonProjError:
                        nonproj_count += 1
                        continue
                    except RootNonProjError:
                        root_nonproj_count += 1
                        continue
                    except CFContainedError:
                        logger.info(f'Cf contained in {sentence_to_str(sentence)}')
                        cfcontained_count += 1
                        continue
                    except NotContainRootError:
                        notcontainroot_count += 1
                        continue

        logger.info(f'Corpus size (sent): {len(corpus)}')
        logger.info(f'Converted sentences: {processed_sentence_num}')
        logger.info(f'and tokens: {token_num}')

        logger.info(f'Non-projective sentences: {nonproj_count}')
        logger.info(f'Root-non-projective sentences: {root_nonproj_count}')
        logger.info(f'Sentences with None: {contain_none_count}')
        logger.info(f'Sentences with empty node: {inclempty_count}')
        logger.info(f'Sentences with control character: {cfcontained_count}')
        logger.info(f'Sentences with not root contained: {notcontainroot_count}')

def remove_data_in_evalset(source_file, G18_conllid_file, tmp_file):
    conllid_set = set()
    with open(G18_conllid_file) as f:
        for conllid in f:
            conllid_set.add(conllid.strip())
    with open(source_file) as f, open(tmp_file, 'w') as g:
        for line in f:
            if line.startswith('#') and line.split(' ')[1] == 'sent_id':
                if line.split(' ')[3].strip() in conllid_set:
                    to_add = False
                else:
                    to_add = True
                    g.write(f'{line}')
            else:
                if to_add:
                    g.write(f'{line}')
    return

if __name__ == '__main__':
    args = parser.parse_args()
    conllu_files_to_convert, method_str, output_dir = generate_path_info(args)

    if args.G18_conllid_file:
        f = tempfile.TemporaryDirectory()
        for conllu_file in conllu_files_to_convert:
            remove_data_in_evalset(conllu_file, args.G18_conllid_file, f'{f.name}/{conllu_file.name}')
        args.source_path = f.name

    conllu_files_to_convert, method_str, output_dir = generate_path_info(args)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    handler = FileHandler(filename=f'{output_dir}/convert.log')
    handler.setLevel(DEBUG)
    handler.setFormatter(Formatter(fmt))
    logger.addHandler(handler)

    convert_conllu_files(args)
