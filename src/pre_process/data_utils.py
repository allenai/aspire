"""
Miscellaneous utilities to read and work with the json files and such.
Stuff multiple functions use.
"""
import sys
import os
import errno

import pandas as pd


class DirIterator:
    def __init__(self, root_path, yield_list, args=None, max_count=None, ):
        """
        Generator over the file names. Typically consumed by the map_unordered
        executable which map_unordered would run.
        :param root_path: string; the directory with the files to iterate over.
        :param yield_list: list; the list of things in_path to yield.
        :param args: tuple; the set of arguments to be returned with each in_file
            and out_file name. This could be the set of arguments which the
            callable consuming the arguments might need. This needs to be fixed
            however.
        :param max_count: int; how many items to yield.
        :returns:
            tuple:
                (in_paper,): a path to a file to open and do things with.
                (in_paper, out_paper): paths to input file and file to write
                    processed content to.
                (in_paper, args): set of arguments to the function
                    doing the processing.
                (in_paper, out_paper, args): set of arguments to the function
                    doing the processing.
        """
        self.in_path = root_path
        self.yield_list = yield_list
        self.optional_args = args
        self.max_count = max_count

    def __iter__(self):
        count = 0
        for doi in self.yield_list:
            if self.max_count:
                if count >= self.max_count:
                    raise StopIteration
            in_paper = os.path.join(self.in_path, doi.strip())
            if self.optional_args:
                yield (in_paper,) + self.optional_args
            else:
                yield in_paper
            count += 1
            

class DirMetaIterator:
    def __init__(self, root_path, yield_list, metadata_df, yield_meta=False, args=None, max_count=None):
        """
        Generator over the file names and yields pids of papers in a file.
        Typically consumed by the map_unordered executable which map_unordered
        would run; specifically consumed by pre_proc_gorc.gather_papers()
        :param root_path: string; the directory with the files to iterate over.
        :param yield_list: list; the list of things in_path to yield.
        :param yield_meta: bool; whether the yielded items should contain parts of metadata_df
        :param metadata_df: pandas.df; metadata data from which to select subsets of
            rows with the same batch id and get pids for.
            29 June 2021: Hack but this can also be a dict.
        :param args: tuple; the set of arguments to be returned with each in_file
            and out_file name. This could be the set of arguments which the
            callable consuming the arguments might need. This needs to be fixed
            however.
        :param max_count: int; how many items to yield.
        :returns:
            tuple:
                (in_paper,): a path to a file to open and do things with.
                (in_paper, out_paper): paths to input file and file to write
                    processed content to.
                (in_paper, args): set of arguments to the function
                    doing the processing.
                (in_paper, out_paper, args): set of arguments to the function
                    doing the processing.
        """
        self.in_path = root_path
        self.yield_list = yield_list
        self.yield_meta = yield_meta
        self.optional_args = args
        self.max_count = max_count
        self.metadata_df = metadata_df

    def __iter__(self):
        count = 0
        for batch_fname in self.yield_list:
            if self.max_count:
                if count >= self.max_count:
                    raise StopIteration
            in_fname = os.path.join(self.in_path, batch_fname.strip())
            if str.endswith(batch_fname, '.jsonl') and str.startswith(batch_fname, 'pid2citcontext-'):
                batch_num = int(batch_fname[15:-6])
            else:
                if str.endswith(batch_fname, 'jsonl.gz'):
                    batch_num = int(batch_fname[:-9])
                elif str.endswith(batch_fname, '.jsonl'):
                    batch_num = int(batch_fname[:-6])
            if isinstance(self.metadata_df, pd.DataFrame):
                batch_metadata_df = self.metadata_df[self.metadata_df['batch_num'] == batch_num]
                pids = set(batch_metadata_df['pid'].values)
            elif isinstance(self.metadata_df, dict):
                pids = set(self.metadata_df[batch_num])
                pids = [int(p) for p in pids]
            if self.yield_meta:
                yield_items = (in_fname, pids, batch_metadata_df)
            else:
                yield_items = (in_fname, pids)
            if self.optional_args:
                yield yield_items + self.optional_args
            else:
                yield yield_items
            count += 1


def create_dir(dir_name):
    """
    Create the directory whose name is passed.
    :param dir_name: String saying the name of directory to create.
    :return: None.
    """
    # Create output directory if it doesnt exist.
    try:
        os.makedirs(dir_name)
        print('Created: {}.'.format(dir_name))
    except OSError as ose:
        # For the case of *file* by name of out_dir existing
        if (not os.path.isdir(dir_name)) and (ose.errno == errno.EEXIST):
            sys.stderr.write('IO ERROR: Could not create output directory\n')
            sys.exit(1)
        # If its something else you don't know; report it and exit.
        if ose.errno != errno.EEXIST:
            sys.stderr.write('OS ERROR: {:d}: {:s}: {:s}\n'.format(ose.errno,
                                                                   ose.strerror,
                                                                   dir_name))
            sys.exit(1)


