import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from os import path, getcwd
from xml.etree import ElementTree as ET

from pkg_resources import resource_filename

sys.path.extend([path.abspath(path.join(getcwd(), path.pardir))])

from rnr_debug_helpers.utils.discovery_wrappers import DiscoveryProxy
from rnr_debug_helpers.utils.io_helpers import initialize_logger, smart_file_open

LOGGER = initialize_logger(logging.INFO, path.basename(__file__))


def _parse_doc_elements(element):
    """
    Parses a single document element from the Solr format xml element into a format Discovery can understand
    :param element:
    :return:
    """
    doc_id, body = None, None
    for field in element.findall("field"):
        if field.attrib['name'] == 'id':
            doc_id = field.text
        elif field.attrib['name'] == 'body':
            body = field.text
    if doc_id is None or body is None:
        raise ValueError('Unable to parse id and body from xml entry: %s' % element)
    return doc_id, {'body': body}


def document_corpus_as_iterable(corpus):
    stats = defaultdict(int)
    with smart_file_open(corpus) as infile:
        LOGGER.info("Loading documents from solr xml file: %s" % corpus)
        # reader = UnicodeRecoder(infile, encoding='utf-8')
        for event, element in ET.iterparse(infile):
            if event == 'end' and element.tag == 'doc':
                stats['num_xml_entries'] += 1
                yield _parse_doc_elements(element)


def main():
    insurance_lib_data_dir = resource_filename('resources', 'insurance_lib_v2')
    print('Using data from {}'.format(insurance_lib_data_dir))

    # Either re-use an existing collection id by over riding the below, or leave as is to create one
    collection_id = "TestCollection-InsLibV2"

    discovery = DiscoveryProxy()

    collection_id = discovery.setup_collection(collection_id=collection_id,
                                               config_id="889a08c9-cad9-4287-a87d-2f0380363bff")
    discovery.print_collection_stats(collection_id)

    # This thing seems to misbehave when run from python notebooks due to its use of multiprocessing, so just
    # running in a script
    discovery.upload_documents(collection_id=collection_id,
                               corpus=document_corpus_as_iterable(
                                   path.join(insurance_lib_data_dir, 'document_corpus.solr.xml')))

    discovery.print_collection_stats(collection_id)


if __name__ == '__main__':
    parser = ArgumentParser(prog="python %s)" % path.basename(__file__),
                            description="Script equivalent of 3.0 - Create Discovery Collection & Evaluate",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--debug', help="Print lots of debugging statements", action="store_const",
                        dest="log_level", const=logging.DEBUG, default=logging.INFO)
    args = parser.parse_args()
    LOGGER = initialize_logger(args.log_level, path.basename(__file__))
    main()
